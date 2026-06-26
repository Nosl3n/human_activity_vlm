# human_kinematic_model.py
# BODY_18 kinematic generator using fixed segment lengths + joint angles.
#
# Optimized parameters (15 kinematic + 1 log-scale = 16 total in state vector):
#   sh_L  axis-angle (3) — Rodrigues rotation for left shoulder
#   sh_R  axis-angle (3) — Rodrigues rotation for right shoulder
#   el_L  flex (1)       — left elbow hinge
#   el_R  flex (1)       — right elbow hinge
#   lb_x  (1)            — lower-body pelvis X translation
#   lb_z  (1)            — lower-body pelvis Z translation
#   lb_roll (1)          — lower-body Y-axis orientation
#   hip_L_flex (1)       — left hip sagittal flexion
#   hip_R_flex (1)       — right hip sagittal flexion
#   knee_L_flex (1)      — left knee flexion
#   knee_R_flex (1)      — right knee flexion
#   log_s (1)            — log of global scale (handled in vfe_inference, not here)
#
# Outputs canonical keypoints (18,3), pelvis/root at origin, Y-up, Z-forward.

from dataclasses import dataclass
from typing import Dict, Tuple
import numpy as np
import torch

KP = {
    "NOSE": 0, "NECK": 1,
    "R_SHOULDER": 2, "R_ELBOW": 3, "R_WRIST": 4,
    "L_SHOULDER": 5, "L_ELBOW": 6, "L_WRIST": 7,
    "R_HIP": 8, "R_KNEE": 9, "R_ANKLE": 10,
    "L_HIP": 11, "L_KNEE": 12, "L_ANKLE": 13,
    "R_EYE": 14, "L_EYE": 15, "R_EAR": 16, "L_EAR": 17,
}

KP18_NAMES = [
    "nose", "neck",
    "r_shoulder", "r_elbow", "r_wrist",
    "l_shoulder", "l_elbow", "l_wrist",
    "r_hip", "r_knee", "r_ankle",
    "l_hip", "l_knee", "l_ankle",
    "r_eye", "l_eye", "r_ear", "l_ear",
]

# ── rotations ─────────────────────────────────────────────────────────────────

def _Rx(a: torch.Tensor) -> torch.Tensor:
    """Rotation about X axis."""
    ca, sa = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([
        torch.stack([o,  z,   z], dim=-1),
        torch.stack([z, ca, -sa], dim=-1),
        torch.stack([z, sa,  ca], dim=-1),
    ], dim=-2)


def _Ry(a: torch.Tensor) -> torch.Tensor:
    """Rotation about Y axis."""
    ca, sa = torch.cos(a), torch.sin(a)
    z, o = torch.zeros_like(a), torch.ones_like(a)
    return torch.stack([
        torch.stack([ ca, z, sa], dim=-1),
        torch.stack([  z, o,  z], dim=-1),
        torch.stack([-sa, z, ca], dim=-1),
    ], dim=-2)


def rodrigues(v: torch.Tensor) -> torch.Tensor:
    """
    Rodrigues axis-angle → rotation matrix.  v: (3,).
    ||v|| is the rotation angle; v/||v|| is the unit axis.
    Avoids gimbal lock and is differentiable at v=0 via clamp.
    """
    theta = v.norm().clamp(min=1e-8)
    k = v / theta
    K = torch.zeros(3, 3, device=v.device, dtype=v.dtype)
    K[0, 1] = -k[2];  K[0, 2] =  k[1]
    K[1, 0] =  k[2];  K[1, 2] = -k[0]
    K[2, 0] = -k[1];  K[2, 1] =  k[0]
    I = torch.eye(3, device=v.device, dtype=v.dtype)
    return I + torch.sin(theta) * K + (1.0 - torch.cos(theta)) * (K @ K)


# ── segment lengths (fixed per subject) ──────────────────────────────────────

@dataclass
class SegmentLengths:
    torso: float
    shoulder_offset_x: float
    hip_offset_x: float
    upper_arm: float
    lower_arm: float
    thigh: float
    calf: float
    # Face offsets relative to NECK (not optimized)
    nose_off: np.ndarray
    reye_off: np.ndarray
    leye_off: np.ndarray
    rear_off: np.ndarray
    lear_off: np.ndarray


def lengths_from_standard_np(standard_kp: np.ndarray) -> SegmentLengths:
    """Derive fixed segment lengths and face offsets from a template skeleton."""
    def d(i, j):
        return float(np.linalg.norm(standard_kp[i] - standard_kp[j]))

    pelvis = 0.5 * (standard_kp[KP["L_HIP"]] + standard_kp[KP["R_HIP"]])
    torso  = float(np.linalg.norm(standard_kp[KP["NECK"]] - pelvis))

    shoulder_offset_x = abs(float(standard_kp[KP["R_SHOULDER"]][0] - standard_kp[KP["NECK"]][0]))
    hip_offset_x      = abs(float(standard_kp[KP["R_HIP"]][0]      - pelvis[0]))

    upper_arm = d(KP["R_SHOULDER"], KP["R_ELBOW"])
    lower_arm = d(KP["R_ELBOW"],    KP["R_WRIST"])
    thigh     = d(KP["R_HIP"],      KP["R_KNEE"])
    calf      = d(KP["R_KNEE"],     KP["R_ANKLE"])

    neck     = standard_kp[KP["NECK"]]
    nose_off = (standard_kp[KP["NOSE"]]  - neck).astype(np.float32)
    reye_off = (standard_kp[KP["R_EYE"]] - neck).astype(np.float32)
    leye_off = (standard_kp[KP["L_EYE"]] - neck).astype(np.float32)
    rear_off = (standard_kp[KP["R_EAR"]] - neck).astype(np.float32)
    lear_off = (standard_kp[KP["L_EAR"]] - neck).astype(np.float32)

    return SegmentLengths(
        torso=torso,
        shoulder_offset_x=shoulder_offset_x,
        hip_offset_x=hip_offset_x,
        upper_arm=upper_arm,
        lower_arm=lower_arm,
        thigh=thigh,
        calf=calf,
        nose_off=nose_off,
        reye_off=reye_off,
        leye_off=leye_off,
        rear_off=rear_off,
        lear_off=lear_off,
    )


# ── kinematic model ───────────────────────────────────────────────────────────

class HumanKinematicModel(torch.nn.Module):
    """
    Forward kinematics for BODY_18 given 15 angle parameters (dict).
    The 16th state parameter (log_s, global scale) is applied in vfe_inference,
    not here, so this module receives and ignores it if present.

    Shoulder rotations use Rodrigues axis-angle (no gimbal lock).
    Lower body has independent hip and knee flexion per leg.
    """

    def __init__(self, lengths: SegmentLengths, device: str = "cpu"):
        super().__init__()
        self.lengths = lengths
        self.device  = device

        self.nose_off = torch.tensor(lengths.nose_off, device=device, dtype=torch.float32)
        self.reye_off = torch.tensor(lengths.reye_off, device=device, dtype=torch.float32)
        self.leye_off = torch.tensor(lengths.leye_off, device=device, dtype=torch.float32)
        self.rear_off = torch.tensor(lengths.rear_off, device=device, dtype=torch.float32)
        self.lear_off = torch.tensor(lengths.lear_off, device=device, dtype=torch.float32)

    def forward(self, angles: Dict[str, torch.Tensor]) -> torch.Tensor:
        L      = self.lengths
        device = self.device
        dtype  = torch.float32

        kp = torch.full((18, 3), float("nan"), device=device, dtype=dtype)

        # ── Lower body global transform ────────────────────────────────────
        lb_x    = angles["lb_x"][0]
        lb_z    = angles["lb_z"][0]
        lb_roll = angles["lb_roll"][0]

        pelvis = torch.stack([lb_x,
                               torch.zeros(1, device=device, dtype=dtype).squeeze(),
                               lb_z])
        R_lb   = _Ry(lb_roll)

        # ── Torso and shoulders ────────────────────────────────────────────
        neck  = pelvis + torch.tensor([0.0, L.torso, 0.0], device=device, dtype=dtype)
        l_sh  = neck   + torch.tensor([-L.shoulder_offset_x, 0.0, 0.0], device=device, dtype=dtype)
        r_sh  = neck   + torch.tensor([ L.shoulder_offset_x, 0.0, 0.0], device=device, dtype=dtype)

        # ── Upper body: Rodrigues for shoulders, hinge for elbows ─────────
        R_sh_L = rodrigues(angles["sh_L"])
        R_sh_R = rodrigues(angles["sh_R"])
        R_el_L = _Rx(angles["el_L"][0])
        R_el_R = _Rx(angles["el_R"][0])

        v_upper = torch.tensor([0.0, -L.upper_arm, 0.0], device=device, dtype=dtype)
        v_lower = torch.tensor([0.0, -L.lower_arm, 0.0], device=device, dtype=dtype)

        l_el = l_sh + R_sh_L @ v_upper
        r_el = r_sh + R_sh_R @ v_upper
        l_wr = l_el + R_sh_L @ (R_el_L @ v_lower)
        r_wr = r_el + R_sh_R @ (R_el_R @ v_lower)

        # ── Lower body: hip positions ──────────────────────────────────────
        l_hip = pelvis + R_lb @ torch.tensor([-L.hip_offset_x, 0.0, 0.0], device=device, dtype=dtype)
        r_hip = pelvis + R_lb @ torch.tensor([ L.hip_offset_x, 0.0, 0.0], device=device, dtype=dtype)

        # ── Per-leg hip + knee flexion (independent) ───────────────────────
        # Hip flexion rotates the thigh in the pelvis-local frame (Rx = sagittal).
        # Knee flexion stacks on the thigh rotation: calf bends relative to thigh.
        # R_lb then maps both into global coordinates.
        R_hip_L  = _Rx(angles["hip_L_flex"][0])
        R_hip_R  = _Rx(angles["hip_R_flex"][0])
        R_knee_L = _Rx(angles["knee_L_flex"][0])
        R_knee_R = _Rx(angles["knee_R_flex"][0])

        v_thigh = torch.tensor([0.0, -L.thigh, 0.0], device=device, dtype=dtype)
        v_calf  = torch.tensor([0.0, -L.calf,  0.0], device=device, dtype=dtype)

        l_kn = l_hip + R_lb @ (R_hip_L @ v_thigh)
        l_an = l_kn  + R_lb @ (R_hip_L @ (R_knee_L @ v_calf))

        r_kn = r_hip + R_lb @ (R_hip_R @ v_thigh)
        r_an = r_kn  + R_lb @ (R_hip_R @ (R_knee_R @ v_calf))

        # ── Face keypoints (fixed offsets from neck) ───────────────────────
        nose  = neck + self.nose_off
        r_eye = neck + self.reye_off
        l_eye = neck + self.leye_off
        r_ear = neck + self.rear_off
        l_ear = neck + self.lear_off

        kp[KP["NOSE"]]       = nose
        kp[KP["NECK"]]       = neck
        kp[KP["R_SHOULDER"]] = r_sh
        kp[KP["R_ELBOW"]]    = r_el
        kp[KP["R_WRIST"]]    = r_wr
        kp[KP["L_SHOULDER"]] = l_sh
        kp[KP["L_ELBOW"]]    = l_el
        kp[KP["L_WRIST"]]    = l_wr
        kp[KP["R_HIP"]]      = r_hip
        kp[KP["R_KNEE"]]     = r_kn
        kp[KP["R_ANKLE"]]    = r_an
        kp[KP["L_HIP"]]      = l_hip
        kp[KP["L_KNEE"]]     = l_kn
        kp[KP["L_ANKLE"]]    = l_an
        kp[KP["R_EYE"]]      = r_eye
        kp[KP["L_EYE"]]      = l_eye
        kp[KP["R_EAR"]]      = r_ear
        kp[KP["L_EAR"]]      = l_ear

        return kp


def default_joint_limits_radians(device: str = "cpu") -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """
    Conservative joint limits for all optimized parameters.

    Shoulders: per-component axis-angle in [-π/2, π/2] (≈ ±90°).
    Elbows:    flexion [0°, 150°].
    Hips:      sagittal flexion [-45°, 120°].
    Knees:     flexion [0°, 150°].
    Pelvis:    ±1 m translation, ±45° roll.
    Scale:     log_s ∈ [-0.5, 0.5]  →  s ∈ [0.61, 1.65].
    """
    def to_rad(deg_list):
        return torch.tensor(deg_list, device=device, dtype=torch.float32) * (torch.pi / 180.0)

    return {
        "sh":      (to_rad([-90.0, -90.0, -90.0]), to_rad([ 90.0,  90.0, 90.0])),
        "el":      (to_rad([  0.0]),                to_rad([150.0])),
        "hip":     (to_rad([-45.0]),                to_rad([120.0])),
        "knee":    (to_rad([  0.0]),                to_rad([150.0])),
        "lb_x":    (torch.tensor([-1.0], device=device, dtype=torch.float32),
                    torch.tensor([ 1.0], device=device, dtype=torch.float32)),
        "lb_z":    (torch.tensor([-1.0], device=device, dtype=torch.float32),
                    torch.tensor([ 1.0], device=device, dtype=torch.float32)),
        "lb_roll": (to_rad([-45.0]), to_rad([45.0])),
        "log_s":   (torch.tensor([-0.5], device=device, dtype=torch.float32),
                    torch.tensor([ 0.5], device=device, dtype=torch.float32)),
    }
