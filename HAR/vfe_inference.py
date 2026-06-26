import numpy as np
import torch
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

from human_kinematic_model import HumanKinematicModel, default_joint_limits_radians


def valid_mask_np(kp: np.ndarray) -> np.ndarray:
    return np.isfinite(kp).all(axis=1)


def kabsch_torch(src: torch.Tensor, dst: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable rigid alignment (rotation + translation, no scale).
    src, dst: (N, 3).  Returns R (3,3), t (3,).
    """
    src_mean = src.mean(dim=0)
    dst_mean = dst.mean(dim=0)
    X = src - src_mean
    Y = dst - dst_mean

    H = X.t() @ Y
    if not torch.isfinite(H).all():
        # Degenerate input — return identity rotation + centroid translation
        return torch.eye(3, device=src.device, dtype=src.dtype), dst_mean - src_mean

    U, S, Vh = torch.linalg.svd(H, full_matrices=False)
    V = Vh.t()

    # Proper rotation (avoid reflection)
    d = torch.sign(torch.linalg.det(V @ U.t()))
    d = torch.where(d == 0, torch.tensor(1.0, device=d.device, dtype=d.dtype), d)
    D = torch.diag(torch.stack([
        torch.ones(1, device=d.device, dtype=d.dtype).squeeze(),
        torch.ones(1, device=d.device, dtype=d.dtype).squeeze(),
        d,
    ]))
    R = V @ D @ U.t()
    t = dst_mean - (src_mean @ R.t())
    return R, t


# ── State vector layout (D = 16) ─────────────────────────────────────────────
#   x[ 0: 3]  sh_L        — left shoulder axis-angle (Rodrigues)
#   x[ 3: 6]  sh_R        — right shoulder axis-angle
#   x[ 6]     el_L        — left elbow flexion
#   x[ 7]     el_R        — right elbow flexion
#   x[ 8]     lb_x        — pelvis X translation
#   x[ 9]     lb_z        — pelvis Z translation
#   x[10]     lb_roll     — lower-body Y rotation
#   x[11]     hip_L_flex  — left hip sagittal flexion
#   x[12]     hip_R_flex  — right hip sagittal flexion
#   x[13]     knee_L_flex — left knee flexion
#   x[14]     knee_R_flex — right knee flexion
#   x[15]     log_s       — log of global skeleton scale (s = exp(log_s))

D_STATE = 16


def _angles_from_vector(x: torch.Tensor) -> Dict[str, torch.Tensor]:
    """
    Unpack the (16,) state vector into an angles dict.
    The kinematic model consumes all keys except 'log_s'.
    """
    i = 0
    sh_L        = x[i:i+3]; i += 3
    sh_R        = x[i:i+3]; i += 3
    el_L        = x[i:i+1]; i += 1
    el_R        = x[i:i+1]; i += 1
    lb_x        = x[i:i+1]; i += 1
    lb_z        = x[i:i+1]; i += 1
    lb_roll     = x[i:i+1]; i += 1
    hip_L_flex  = x[i:i+1]; i += 1
    hip_R_flex  = x[i:i+1]; i += 1
    knee_L_flex = x[i:i+1]; i += 1
    knee_R_flex = x[i:i+1]; i += 1
    log_s       = x[i:i+1]; i += 1
    return {
        "sh_L": sh_L, "sh_R": sh_R,
        "el_L": el_L, "el_R": el_R,
        "lb_x": lb_x, "lb_z": lb_z, "lb_roll": lb_roll,
        "hip_L_flex": hip_L_flex, "hip_R_flex": hip_R_flex,
        "knee_L_flex": knee_L_flex, "knee_R_flex": knee_R_flex,
        "log_s": log_s,
    }


def symmetry_prior(angles: Dict[str, torch.Tensor]) -> torch.Tensor:
    """
    Soft left/right symmetry for the full body:
    - Shoulders (axis-angle): x-component equal, y/z components mirrored.
    - Elbows: equal flexion.
    - Hips: equal flexion (symmetric gait/stance).
    - Knees: equal flexion.
    """
    shL, shR = angles["sh_L"], angles["sh_R"]
    elL, elR = angles["el_L"], angles["el_R"]

    sh_err   = (shL[0] - shR[0])**2 + (shL[1] + shR[1])**2 + (shL[2] + shR[2])**2
    el_err   = (elL[0] - elR[0])**2
    hip_err  = (angles["hip_L_flex"][0]  - angles["hip_R_flex"][0])**2
    knee_err = (angles["knee_L_flex"][0] - angles["knee_R_flex"][0])**2

    return sh_err + el_err + hip_err + knee_err


def joint_limits_prior(x: torch.Tensor,
                       lim: Dict[str, Tuple[torch.Tensor, torch.Tensor]]) -> torch.Tensor:
    """Soft penalty: relu(min - v)^2 + relu(v - max)^2 for each parameter."""
    angles = _angles_from_vector(x)

    def penalty(v, vmin, vmax):
        return torch.relu(vmin - v).pow(2).sum() + torch.relu(v - vmax).pow(2).sum()

    sh_min,      sh_max      = lim["sh"]
    el_min,      el_max      = lim["el"]
    hip_min,     hip_max     = lim["hip"]
    knee_min,    knee_max    = lim["knee"]
    lb_x_min,    lb_x_max    = lim["lb_x"]
    lb_z_min,    lb_z_max    = lim["lb_z"]
    lb_roll_min, lb_roll_max = lim["lb_roll"]
    log_s_min,   log_s_max   = lim["log_s"]

    p = torch.tensor(0.0, device=x.device, dtype=x.dtype)
    p = p + penalty(angles["sh_L"],        sh_min,      sh_max)
    p = p + penalty(angles["sh_R"],        sh_min,      sh_max)
    p = p + penalty(angles["el_L"],        el_min,      el_max)
    p = p + penalty(angles["el_R"],        el_min,      el_max)
    p = p + penalty(angles["hip_L_flex"],  hip_min,     hip_max)
    p = p + penalty(angles["hip_R_flex"],  hip_min,     hip_max)
    p = p + penalty(angles["knee_L_flex"], knee_min,    knee_max)
    p = p + penalty(angles["knee_R_flex"], knee_min,    knee_max)
    p = p + penalty(angles["lb_x"],        lb_x_min,    lb_x_max)
    p = p + penalty(angles["lb_z"],        lb_z_min,    lb_z_max)
    p = p + penalty(angles["lb_roll"],     lb_roll_min, lb_roll_max)
    p = p + penalty(angles["log_s"],       log_s_min,   log_s_max)
    return p


@dataclass
class InferenceConfig:
    anchors: List[int]
    device: str = "cpu"

    # Observation noise base (used when no confidence provided)
    sigma_obs: float = 0.06

    # Dynamics prior noise (how fast angles can change frame-to-frame)
    sigma_dyn: float = 0.25

    # Precision mapping from keypoint confidence
    sigma_min: float = 0.02
    sigma_max: float = 0.15

    # Prior weights
    w_limits: float = 5.0
    w_sym:    float = 1.0

    # Gauss–Newton / Laplace settings
    gn_steps: int   = 2
    damping:  float = 1e-3


@dataclass
class InferenceResult:
    mu: np.ndarray              # (16,) posterior mean  [angles + log_s]
    Lambda: np.ndarray          # (16,16) posterior precision
    kp_pred_aligned: np.ndarray # (18,3) prediction aligned to live frame
    R: np.ndarray               # (3,3)  Kabsch rotation
    t: np.ndarray               # (3,)   Kabsch translation
    diff: np.ndarray            # (18,3) live − pred residuals
    per_l2: np.ndarray          # (18,)  per-joint L2 error
    mean_l2: float
    rmse: float
    used_anchors: List[int]
    valid_count: int
    uncertainty_trace: float    # trace(Σ) as uncertainty proxy


class AInfLaplacePoseEstimator:
    """
    Active Inference with Laplace approximation.

    Maintains belief q(θ) ~ N(μ, Σ) over 16 parameters:
      - 15 kinematic angles (shoulders via Rodrigues, elbows, hips, knees, pelvis)
      - 1 log-scale parameter (adapts template size to subject)

    Each frame updates μ by minimising variational free energy F with a
    Gauss–Newton step:  δ = H⁻¹ g,  μ ← μ − δ
    where H ≈ J^T J + Λ_dyn + damping·I,  g = ∇F.
    """

    def __init__(self, model: HumanKinematicModel, cfg: InferenceConfig):
        self.model = model
        self.cfg   = cfg
        self.lim   = default_joint_limits_radians(device=cfg.device)

        self.mu     = torch.zeros(D_STATE, device=cfg.device, dtype=torch.float32)
        self.Lambda = torch.eye(D_STATE,  device=cfg.device, dtype=torch.float32)

    def _joint_precisions(self, valid_mask: np.ndarray,
                          kp_conf: Optional[np.ndarray],
                          device: str) -> torch.Tensor:
        """Per-valid-joint precision π (M,)."""
        if kp_conf is None:
            return torch.ones(int(valid_mask.sum()), device=device,
                              dtype=torch.float32) * (1.0 / self.cfg.sigma_obs ** 2)

        confv = np.clip(kp_conf.astype(np.float32)[valid_mask], 0.0, 100.0) / 100.0
        sigma = self.cfg.sigma_min + (self.cfg.sigma_max - self.cfg.sigma_min) * (1.0 - confv)
        return torch.tensor(1.0 / sigma ** 2, device=device, dtype=torch.float32)

    def _predict_aligned(self, mu_var: torch.Tensor,
                         live: torch.Tensor,
                         anchors: List[int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Canonical kp → scale → Kabsch → aligned prediction.
        Returns (pred_aligned, R, t).
        """
        angles   = _angles_from_vector(mu_var)
        kp_pred  = self.model(angles)                        # (18,3) canonical
        s        = torch.exp(angles["log_s"][0])             # global scale
        kp_scaled = kp_pred * s
        R, t     = kabsch_torch(kp_scaled[anchors], live[anchors])
        pred_aligned = kp_scaled @ R.t() + t
        return pred_aligned, R, t

    def infer(self, live_kp_np: np.ndarray,
              kp_conf_np: Optional[np.ndarray] = None) -> InferenceResult:
        device = self.cfg.device
        live   = torch.tensor(live_kp_np, dtype=torch.float32, device=device)

        valid       = valid_mask_np(live_kp_np)
        valid_count = int(valid.sum())

        # Choose anchors that are currently valid
        anchors = [i for i in self.cfg.anchors if valid[i]]
        if len(anchors) < 3:
            anchors = [i for i in range(18) if valid[i]]
        if len(anchors) < 3:
            nan = np.full((18, 3), np.nan, dtype=np.float32)
            return InferenceResult(
                mu=self.mu.detach().cpu().numpy(),
                Lambda=self.Lambda.detach().cpu().numpy(),
                kp_pred_aligned=nan,
                R=np.eye(3, dtype=np.float32),
                t=np.zeros(3, dtype=np.float32),
                diff=nan,
                per_l2=np.full((18,), np.nan, dtype=np.float32),
                mean_l2=float("nan"),
                rmse=float("nan"),
                used_anchors=anchors,
                valid_count=valid_count,
                uncertainty_trace=float("inf"),
            )

        # Dynamics prior: θ_t ~ N(θ_{t−1}, σ_dyn² I)
        mu_prior   = self.mu.clone()
        Lambda_dyn = torch.eye(D_STATE, device=device, dtype=torch.float32) * \
                     (1.0 / self.cfg.sigma_dyn ** 2)

        # Per-joint precision (valid joints only)
        pi = self._joint_precisions(valid, kp_conf_np, device=device)  # (M,)

        # Gauss–Newton / Laplace iterations
        for _ in range(self.cfg.gn_steps):
            mu_var = self.mu.clone().detach().requires_grad_(True)

            pred_aligned, R, t = self._predict_aligned(mu_var, live, anchors)
            diff = live - pred_aligned                          # (18,3)

            vmask_t = torch.tensor(valid, device=device)
            diff_v  = diff[vmask_t]                            # (M,3)

            w = torch.sqrt(pi).unsqueeze(1)                    # (M,1)
            r = (w * diff_v).reshape(-1)                       # (M*3,)

            F_like = 0.5 * (r @ r) / max(1.0, float(pi.numel()))

            dmu   = mu_var - mu_prior
            F_dyn = 0.5 * (dmu[None, :] @ Lambda_dyn @ dmu[:, None]).squeeze()

            angles = _angles_from_vector(mu_var)
            F_sym  = symmetry_prior(angles)
            F_lim  = joint_limits_prior(mu_var, self.lim)

            F = F_like + F_dyn + self.cfg.w_sym * F_sym + self.cfg.w_limits * F_lim

            g = torch.autograd.grad(F, mu_var, create_graph=False)[0]  # (D,)

            def r_of(z):
                predA, _, _ = self._predict_aligned(z, live, anchors)
                d = (live - predA)[vmask_t]
                return (w * d).reshape(-1)

            # Jacobian shape: (M*3, D_STATE)
            J = torch.autograd.functional.jacobian(r_of, mu_var, create_graph=False)

            H_like  = (J.t() @ J) / max(1.0, float(pi.numel()))
            H_total = H_like + Lambda_dyn + \
                      (self.cfg.damping * torch.eye(D_STATE, device=device))

            delta = torch.linalg.solve(H_total, g)

            # Skip update if solve produced non-finite values
            if torch.isfinite(delta).all():
                self.mu = mu_var.detach() - delta.detach()

            self.Lambda = H_total.detach()

        # Final forward pass for output
        with torch.no_grad():
            pred_aligned, R, t = self._predict_aligned(self.mu, live, anchors)
            diff = live - pred_aligned

            diff_np = diff.cpu().numpy()
            pred_np = pred_aligned.cpu().numpy()

            per_l2 = np.full((18,), np.nan, dtype=np.float32)
            for i in range(18):
                if valid[i]:
                    per_l2[i] = float(np.linalg.norm(diff_np[i]))

            vals    = per_l2[np.isfinite(per_l2)]
            mean_l2 = float(np.mean(vals))             if vals.size else float("nan")
            rmse    = float(np.sqrt(np.mean(vals**2))) if vals.size else float("nan")

            try:
                cov               = torch.linalg.inv(self.Lambda)
                uncertainty_trace = float(torch.trace(cov).item())
            except Exception:
                uncertainty_trace = float("inf")

        return InferenceResult(
            mu=self.mu.detach().cpu().numpy(),
            Lambda=self.Lambda.detach().cpu().numpy(),
            kp_pred_aligned=pred_np,
            R=R.cpu().numpy(),
            t=t.cpu().numpy(),
            diff=diff_np,
            per_l2=per_l2,
            mean_l2=mean_l2,
            rmse=rmse,
            used_anchors=anchors,
            valid_count=valid_count,
            uncertainty_trace=uncertainty_trace,
        )
