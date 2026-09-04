#!/usr/bin/env python3
"""
trust_metric.py -- the communication-aware trust metric for twin_fed.py.

Implements BOTH exposure scalarizations from a single spectrum computation:

    lmax  :  u = w^2 * nu_l * ( lam_max((G+eI)^-1) - lam_max((G_inf+eI)^-1) )_+
             = w^2 * nu_l * ( 1/(lam_min(G)+e) - L0 )_+
             -> the form the paper's theory is stated for (Prop. 6 ceiling,
                Eq. (16) feasibility). Empirically BINARY on the 123-node
                feeder: it is a hard observability-loss alarm.

    trace :  u = w^2 * nu_t * ( tr((G+eI)^-1) - tr((G_inf+eI)^-1) )_+
             -> graded: sums 1/(lam_i+e) over ALL n directions, so it responds
                to partial degradation and to age, which lam_max does not.

Both share EXACTLY the same g_i (Lemma 1), the same age-aware estimator
(Eq. 5), and the same residual r (Eq. 12). They differ only in how the
information matrix is scalarized, so any ROC difference between them is
attributable to the scalarization alone.

Theory that transfers, and theory that does not
-----------------------------------------------
  P1 range, P2 residual monotonicity, P3 communication monotonicity,
  P5 reduction, Cor. 1 (chi^2)          -- hold for BOTH (trace is
                                           Loewner-monotone, like lam_max).
  P6 ceiling / Eq. (16) feasibility     -- DIFFERENT constants for trace;
                                           both derived and checked below.
  P8 delta-confident floor              -- generalised: Weyl bounds EVERY
                                           eigenvalue by the same s_delta, so
                                           the floor is applied spectrum-wise
                                           and the trace bound follows.

Cost: one eigvalsh per update (~14 ms at n=491) serves all four exposures.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np
from scipy.stats import binom

__all__ = ["MetricConfig", "TrustMetric"]


# --------------------------------------------------------------------- config
@dataclass
class MetricConfig:
    # --- THE TOGGLE: which exposure drives s(t), T(t) and the main AUCs ---
    scalarization: str = "lmax"        # "lmax" | "trace"
    exposure_form: str = "mean"        # "mean" | "delta"  (Prop. 8)

    eps: float = 5.688e-3              # MUST satisfy Eq. (16); see check_feasibility
    omega: float = 0.02                # per-update state increment bound, p.u.
    r0: float = 1.0                    # residual normalizer (shared)
    u0_lmax: float = 1.0               # exposure normalizer, lmax form
    u0_trace: float = 1.0              # exposure normalizer, trace form
    T_th: float = 0.70                 # alarm threshold
    delta: float = 0.05                # confidence level for the delta form
    b_min: float = 0.0                 # bandwidth feasibility threshold

    n_telemetry: Optional[int] = None  # first n_tel channels are starvable;
                                       # the rest are pseudo (Delta=0, g=1)
    ridge: float = 0.0                 # optional estimator ridge if rank-deficient

    def u0(self) -> float:
        return self.u0_lmax if self.scalarization == "lmax" else self.u0_trace


# ---------------------------------------------------------------- the metric
class TrustMetric:
    """Precomputes everything that does not change between updates."""

    def __init__(self, H, sigma2, Q, cfg: MetricConfig):
        self.H = np.asarray(H, float)
        self.s2 = np.asarray(sigma2, float)
        self.cfg = cfg
        self.m, self.n = self.H.shape

        # Lemma 1: tau_i = sigma_i^2 / (h_i' Q h_i); guard h'Qh == 0 -> g_i == 1
        hQh = np.einsum("ij,jk,ik->i", self.H, np.asarray(Q, float), self.H)
        self.tau = np.where(hQh > 0, self.s2 / np.maximum(hQh, 1e-300), np.inf)

        # rank-1 helpers (never form the m x n x n tensor)
        self.h2 = np.einsum("ij,ij->i", self.H, self.H)
        self.Kn = self.h2 / self.s2                    # ||K_i||_2, exact
        self._Kn_sorted = np.sort(self.Kn)[::-1]

        # perfect-telemetry spectrum and both baselines
        lam_inf = self._spectrum(np.ones(self.m))
        self.lam_inf = lam_inf
        self.L0 = 1.0 / (lam_inf[0] + cfg.eps)                 # lam_max((G_inf+eI)^-1)
        self.T0 = float(np.sum(1.0 / (lam_inf + cfg.eps)))     # tr((G_inf+eI)^-1)
        self.nu_l = 1.0 / self.L0
        self.nu_t = 1.0 / self.T0

        # ceilings (Prop. 6 and its trace analogue)
        self.u_max_lmax = cfg.omega**2 * lam_inf[0] / cfg.eps
        self.u_max_trace = cfg.omega**2 * (self.n / (cfg.eps * self.T0) - 1.0)

        # achievable maxima with pseudo-measurements always present
        if cfg.n_telemetry:
            w = np.ones(self.m); w[:cfg.n_telemetry] = 0.0
            lam_p = self._spectrum(w)
            self.u_ach_lmax = cfg.omega**2 * self.nu_l * max(
                1.0 / (lam_p[0] + cfg.eps) - self.L0, 0.0)
            self.u_ach_trace = cfg.omega**2 * self.nu_t * max(
                float(np.sum(1.0 / (lam_p + cfg.eps))) - self.T0, 0.0)
        else:
            self.u_ach_lmax, self.u_ach_trace = self.u_max_lmax, self.u_max_trace

    # ------------------------------------------------------------- internals
    def _spectrum(self, w) -> np.ndarray:
        """ascending eigenvalues of G(w) = sum_i w_i K_i, via SVD of the
        row-scaled matrix (eigvalsh of G returns NEGATIVE values at kappa~1e9)."""
        w = np.maximum(np.asarray(w, float), 0.0)
        A = self.H * np.sqrt(w / self.s2)[:, None]
        sv = np.linalg.svd(A, compute_uv=False)
        lam = np.zeros(self.n)
        k = min(len(sv), self.n)
        lam[:k] = sv[:k] ** 2
        return np.sort(lam)

    def g_of(self, Delta) -> np.ndarray:
        """Lemma 1 freshness discount. Pseudo-measurements never age."""
        D = np.asarray(Delta, float).copy()
        if self.cfg.n_telemetry:
            D[self.cfg.n_telemetry:] = 0.0
        out = np.ones(self.m)
        fin = np.isfinite(self.tau)
        out[fin] = 1.0 / (1.0 + D[fin] / self.tau[fin])
        return out

    def rho_of(self, p, Delta, b) -> np.ndarray:
        """Eq. (2): rho_i = (1-p_i) g_i(Delta_i) 1[b_i >= b_min]."""
        g = self.g_of(Delta)
        feas = (np.asarray(b, float) >= self.cfg.b_min).astype(float)
        return (1.0 - np.asarray(p, float)) * g * feas, g

    def _delta_floor(self, p, gamma) -> np.ndarray:
        """Prop. 8 generalised: apply the deflation SPECTRUM-WISE.

        Weyl gives lam_i(G_real) >= lam_i(G) - s_delta for EVERY i, and the
        same for the loss-quantile constituent, so both the lam_max and the
        trace bounds follow from one floored spectrum. Envelopes are the
        gamma-free, p-monotone ones (Eq. 10) -- do NOT reintroduce the tight
        gamma-weighted forms, they are non-monotone.
        """
        cfg = self.cfg
        p = np.asarray(p, float)
        lam = self._spectrum((1.0 - p) * gamma)
        phi = np.where(p <= 0.5, p * (1.0 - p), 0.25)          # nondecreasing in p
        Vbar = float(np.max(self._spectrum(phi * self.Kn)))    # ||sum phi_i K_i^2||
        stoch = self.Kn[p > 0]
        Lbar = float(stoch.max()) if stoch.size else 0.0       # gamma-free
        lg = np.log(4.0 * self.n / cfg.delta)
        s_d = np.sqrt(2.0 * Vbar * lg) + (2.0 * Lbar / 3.0) * lg
        f_bern = np.maximum(lam - s_d, 0.0)

        k = int(binom.ppf(1.0 - cfg.delta / 2.0, self.m, float(p.mean()))) if p.max() > 0 else 0
        Lam = float(self._Kn_sorted[:k].sum()) if k else 0.0   # UNWEIGHTED norms
        f_quant = np.maximum(self._spectrum(gamma) - Lam, 0.0)
        return np.maximum(f_bern, f_quant)                      # combined, Eq. (11)

    # ------------------------------------------------------------------ API
    def estimate(self, z, rx, gamma):
        """Solve the age-aware WLS problem without forming normal equations.

        Rows are scaled by sqrt(gamma / sigma2), and the resulting
        least-squares system is solved directly by rank-revealing SVD.
        The Boolean return value means full numerical column rank and
        a finite solution; a LAPACK call merely returning is not enough.
        """
        z_array = np.asarray(z, dtype=float).reshape(-1)
        rx_array = np.asarray(rx, dtype=bool).reshape(-1)
        gamma_array = np.asarray(gamma, dtype=float).reshape(-1)

        if (
            len(z_array) != self.m
            or len(rx_array) != self.m
            or len(gamma_array) != self.m
        ):
            raise ValueError(
                "z, rx, and gamma must match the measurement dimension"
            )

        idx = np.flatnonzero(rx_array)
        raw_weights = gamma_array[idx] / self.s2[idx]

        valid = (
            np.isfinite(raw_weights)
            & (raw_weights > 0.0)
            & np.isfinite(z_array[idx])
        )

        idx = idx[valid]
        weights = raw_weights[valid]

        rcond = float(np.sqrt(np.finfo(float).eps))
        ridge = float(self.cfg.ridge)

        if not np.isfinite(ridge) or ridge < 0.0:
            raise ValueError(
                "estimator ridge must be finite and nonnegative"
            )

        self.last_estimator_solver = "weighted_lstsq_svd"
        self.last_estimator_rcond = rcond
        self.last_estimator_effective_rows = int(len(idx))
        self.last_estimator_rank = 0
        self.last_estimator_condition = float("inf")
        self.last_estimator_singular_max = float("nan")
        self.last_estimator_singular_min = float("nan")
        self.last_estimator_residual_norm = float("nan")

        if len(idx) == 0:
            return np.full(self.n, np.nan), False

        square_root_weights = np.sqrt(weights)
        weighted_h = (
            self.H[idx]
            * square_root_weights[:, None]
        )
        weighted_z = (
            z_array[idx]
            * square_root_weights
        )

        if (
            not np.all(np.isfinite(weighted_h))
            or not np.all(np.isfinite(weighted_z))
        ):
            return np.full(self.n, np.nan), False

        if ridge > 0.0:
            weighted_h = np.vstack(
                [
                    weighted_h,
                    np.eye(self.n) * np.sqrt(ridge),
                ]
            )
            weighted_z = np.concatenate(
                [
                    weighted_z,
                    np.zeros(self.n),
                ]
            )

        try:
            (
                candidate,
                _,
                rank,
                singular_values,
            ) = np.linalg.lstsq(
                weighted_h,
                weighted_z,
                rcond=rcond,
            )
        except np.linalg.LinAlgError:
            return np.full(self.n, np.nan), False

        candidate = np.asarray(
            candidate,
            dtype=float,
        ).reshape(-1)

        rank = int(rank)
        singular_values = np.asarray(
            singular_values,
            dtype=float,
        )

        if len(singular_values) > 0:
            singular_max = float(
                singular_values[0]
            )
            singular_min = float(
                singular_values[-1]
            )

            condition = (
                float(singular_max / singular_min)
                if singular_min > 0.0
                else float("inf")
            )
        else:
            singular_max = float("nan")
            singular_min = float("nan")
            condition = float("inf")

        candidate_finite = bool(
            candidate.shape == (self.n,)
            and np.all(np.isfinite(candidate))
        )

        residual_norm = (
            float(
                np.linalg.norm(
                    weighted_h @ candidate
                    - weighted_z
                )
            )
            if candidate_finite
            else float("nan")
        )

        full_numerical_rank = bool(
            rank == self.n
            and np.isfinite(condition)
            and condition <= 1.0 / rcond
        )

        self.last_estimator_rank = rank
        self.last_estimator_condition = condition
        self.last_estimator_singular_max = singular_max
        self.last_estimator_singular_min = singular_min
        self.last_estimator_residual_norm = residual_norm

        solved_reliably = bool(
            candidate_finite
            and full_numerical_rank
        )

        return candidate, solved_reliably

    def update(self, p, Delta, b, z, rx=None, x_hat=None) -> dict:
        """One twin update. Returns BOTH exposures plus the active s(t), T(t)."""
        cfg = self.cfg
        rho, gamma = self.rho_of(p, Delta, b)
        if rx is None:                       # received AND current
            rx = (np.asarray(p, float) < 1.0) & (np.asarray(b, float) >= cfg.b_min)
        rx = np.asarray(rx, bool)

        observable = True
        if x_hat is None:
            x_hat, observable = self.estimate(z, rx, gamma)

        # Eq. (12): g-weighted, normalized by TOTAL m. E[r] = (|M_rx| - n)/m
        idx = np.flatnonzero(rx)
        resid = np.asarray(z, float)[idx] - self.H[idx] @ x_hat
        r = float(np.sum(gamma[idx] * resid**2 / self.s2[idx]) / self.m)

        # one spectrum serves every exposure
        lam_mean = self._spectrum(rho)
        lam_dlt = self._delta_floor(p, gamma) if cfg.exposure_form == "delta" else None
        lam_use = lam_dlt if lam_dlt is not None else lam_mean

        u_l = cfg.omega**2 * self.nu_l * max(1.0 / (lam_use[0] + cfg.eps) - self.L0, 0.0)
        u_t = cfg.omega**2 * self.nu_t * max(
            float(np.sum(1.0 / (lam_use + cfg.eps))) - self.T0, 0.0)
        # always report the mean form too, so a delta run stays comparable
        u_l_mean = cfg.omega**2 * self.nu_l * max(1.0 / (lam_mean[0] + cfg.eps) - self.L0, 0.0)
        u_t_mean = cfg.omega**2 * self.nu_t * max(
            float(np.sum(1.0 / (lam_mean + cfg.eps))) - self.T0, 0.0)

        u = u_l if cfg.scalarization == "lmax" else u_t
        s = r / cfg.r0 + u / cfg.u0()
        T = float(np.exp(-s))

        n_rx = int(rx.sum())
        n_tel = cfg.n_telemetry or self.m
        return dict(
            T=T, s=s, r=r, u=u, alarm=bool(T < cfg.T_th),
            u_lmax=u_l, u_trace=u_t, u_lmax_mean=u_l_mean, u_trace_mean=u_t_mean,
            s_lmax=r / cfg.r0 + u_l / cfg.u0_lmax,
            s_trace=r / cfg.r0 + u_t / cfg.u0_trace,
            lam_min=float(lam_use[0]), trace_inv=float(np.sum(1.0 / (lam_use + cfg.eps))),
            n_rx=n_rx, x_hat=x_hat, observable=observable,
            # trivial controls B1/B2 over the STARVABLE set only
            b1=float(rx[:n_tel].mean()),
            b2=float(np.mean(np.asarray(Delta, float)[:n_tel])),
            E_r_expected=(n_rx - self.n) / self.m,
        )

    # --------------------------------------------------------- housekeeping
    def check_feasibility(self, verbose=True) -> dict:
        """Eq. (16) for lmax and its trace analogue. Both must hold for the
        chosen scalarization or the metric cannot alarm under total starvation."""
        cfg = self.cfg
        lg = np.log(1.0 / cfg.T_th)
        need_l = cfg.omega**2 * self.lam_inf[0] / (cfg.u0_lmax * lg)
        # trace: u_max_tr/u0 > ln(1/T_th)  =>  eps < n / (T0 * (1 + u0*ln/w^2))
        need_t = self.n / (self.T0 * (1.0 + cfg.u0_trace * lg / cfg.omega**2))
        out = dict(eps=cfg.eps, eps_max_lmax=float(need_l), eps_max_trace=float(need_t),
                   ok_lmax=cfg.eps < need_l, ok_trace=cfg.eps < need_t,
                   u_max_lmax=self.u_max_lmax, u_max_trace=self.u_max_trace,
                   u_achievable_lmax=self.u_ach_lmax, u_achievable_trace=self.u_ach_trace)
        if verbose:
            print(f"  eps = {cfg.eps:.6g}")
            print(f"    lmax : limit {need_l:.6g}  -> {'OK' if out['ok_lmax'] else 'VIOLATED'}"
                  f"   ceiling {self.u_max_lmax:.4g}   achievable {self.u_ach_lmax:.4g}")
            print(f"    trace: limit {need_t:.6g}  -> {'OK' if out['ok_trace'] else 'VIOLATED'}"
                  f"   ceiling {self.u_max_trace:.4g}   achievable {self.u_ach_trace:.4g}")
            # is the delta-confident form usable at all on this feeder?
            if cfg.n_telemetry:
                pp = np.zeros(self.m); pp[:cfg.n_telemetry] = 0.05
                _, gg = self.rho_of(pp, np.zeros(self.m), np.full(self.m, np.inf))
                fl = self._delta_floor(pp, gg)
                dead = int(np.sum(fl <= 0))
                if dead > 0.5 * self.n:
                    print(f"    [WARN] delta-confident floor collapses {dead}/{self.n} "
                          f"directions at p=0.05:\n           u^delta is VACUOUS here. Use "
                          f"exposure_form='mean' as primary and report\n           the delta "
                          f"form only as a conservative check.")
            if self.u_ach_lmax < 0.5 * self.u_max_lmax:
                print("    note: pseudo-measurements keep G away from 0, so the ACHIEVABLE"
                      "\n          exposure is below the theoretical ceiling. Calibrate u0"
                      "\n          against achievable values, not against the ceiling.")
        return out

    def calibrate_u0(self, patterns, target_frac=1.0, which="both"):
        """Set u0 so the given (p, Delta, b) boundary patterns land at T_th.
        u0 = u_boundary / (ln(1/T_th) - r/r0). Calibrate on the ORACLE,
        ample-comms regime only -- never on the evaluation set."""
        cfg = self.cfg
        lg = np.log(1.0 / cfg.T_th)
        ul, ut = [], []
        for (p, D, b) in patterns:
            rho, gamma = self.rho_of(p, D, b)
            lam = self._spectrum(rho)
            ul.append(cfg.omega**2 * self.nu_l * max(1.0 / (lam[0] + cfg.eps) - self.L0, 0.0))
            ut.append(cfg.omega**2 * self.nu_t * max(
                float(np.sum(1.0 / (lam + cfg.eps))) - self.T0, 0.0))
        if which in ("both", "lmax") and np.median(ul) > 0:
            cfg.u0_lmax = float(np.median(ul) / (target_frac * lg))
        if which in ("both", "trace") and np.median(ut) > 0:
            cfg.u0_trace = float(np.median(ut) / (target_frac * lg))
        return cfg.u0_lmax, cfg.u0_trace


# ------------------------------------------------------------------ self-test
if __name__ == "__main__":
    import sys, json
    d = np.load(sys.argv[1] if len(sys.argv) > 1 else "feeder.npz", allow_pickle=True)
    H, s2, Q = d["H"], d["sigma2"], d["Q"]
    nt = int(d["n_telemetry"])
    cfg = MetricConfig(n_telemetry=nt)
    tm = TrustMetric(H, s2, Q, cfg)
    m = tm.m
    print("=" * 70); print("trust_metric.py self-test"); print("=" * 70)
    print(f"  m={m}  n={tm.n}  telemetry={nt}")
    tm.check_feasibility()

    rng = np.random.default_rng(0)
    z0 = H @ (rng.standard_normal(tm.n) * 0.01)
    zero, big = np.zeros(m), np.full(m, 1e9)

    print("\n  P5 reduction (perfect telemetry): u must vanish for BOTH")
    out = tm.update(zero, zero, big, z0)
    print(f"    u_lmax={out['u_lmax']:.3e}   u_trace={out['u_trace']:.3e}"
          f"   -> {'PASS' if max(out['u_lmax'], out['u_trace']) < 1e-12 else 'FAIL'}")

    print("\n  age response (all telemetry aged, no loss):")
    for D0 in (0.0, 1.0, 10.0, 60.0):
        D = np.zeros(m); D[:nt] = D0
        o = tm.update(zero, D, big, z0)
        print(f"    Delta={D0:5.1f}s  u_lmax={o['u_lmax']:.4e}  u_trace={o['u_trace']:.4e}")

    print("\n  loss response (k telemetry lost):")
    for k in (5, 20, 45):
        p = np.zeros(m); p[rng.choice(nt, k, replace=False)] = 1.0
        o = tm.update(p, zero, big, z0)
        print(f"    k={k:2d}  u_lmax={o['u_lmax']:.4e}  u_trace={o['u_trace']:.4e}"
              f"  n_rx={o['n_rx']}  b1={o['b1']:.3f}")

    print("\n  monotonicity of BOTH exposures under worsening loss:")
    prev_l = prev_t = -np.inf; viol = 0
    for pv in (0.0, 0.05, 0.1, 0.2, 0.4, 0.6, 0.8, 1.0):
        p = np.zeros(m); p[:nt] = pv
        o = tm.update(p, zero, big, z0)
        if o["u_lmax"] < prev_l - 1e-12 or o["u_trace"] < prev_t - 1e-12:
            viol += 1
        prev_l, prev_t = o["u_lmax"], o["u_trace"]
    print(f"    violations={viol}  -> {'PASS' if viol == 0 else 'FAIL'}")

    print("\n  delta-confident form (spectrum-wise floor):")
    cfg.exposure_form = "delta"
    p = np.zeros(m); p[:nt] = 0.2
    o = tm.update(p, zero, big, z0)
    print(f"    u_lmax={o['u_lmax']:.4e} (mean {o['u_lmax_mean']:.4e})"
          f"   u_trace={o['u_trace']:.4e} (mean {o['u_trace_mean']:.4e})")
    cfg.exposure_form = "mean"

    print("\n  toggle check: cfg.scalarization drives s(t)/T(t)")
    for sc in ("lmax", "trace"):
        cfg.scalarization = sc
        o = tm.update(p, zero, big, z0)
        print(f"    {sc:5s}: u={o['u']:.4e}  s={o['s']:.4f}  T={o['T']:.4f}"
              f"  alarm={o['alarm']}")
