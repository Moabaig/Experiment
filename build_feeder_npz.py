#!/usr/bin/env python3
"""
build_feeder_npz.py — produce a REAL feeder.npz for preflight.py from the
authentic IEEE 123-node test feeder (OpenDSS), not a synthetic surrogate.

Outputs H (m x n), sigma2 (m,), Q (n x n) plus provenance metadata.

State  x = [ theta (all nodes except 3 slack phase-references) ; |V| (all nodes) ]
       per unit, so n = 2*N_nodes - 3.
Jacobian rows are exact analytic derivatives of the multiphase nodal power
injections / flows, computed from the OpenDSS Ybus at the solved point.

Channel set = TELEMETRY ONLY (things that traverse the comms network and can
therefore be starved).  Pseudo-measurements are model priors, do not traverse
the network, and are exported separately as H_pseudo so they can be added as a
prior rather than treated as starvable channels.
"""
import os, json, numpy as np, scipy.sparse as sp
import opendssdirect as dss

MASTER = os.path.abspath("IEEE123Master.dss")
rng = np.random.default_rng(0)

# ------------------------------------------------------------------ solve
dss.Text.Command(f'Redirect "{MASTER}"')
dss.Text.Command("Set Maxiterations=100")
dss.Solution.Solve()
assert dss.Solution.Converged(), "power flow did not converge"

order = list(dss.Circuit.YNodeOrder())
N = len(order)
Vc = np.array(dss.Circuit.YNodeVArray()).view(complex) if False else \
     np.array(dss.Circuit.YNodeVArray())[0::2] + 1j*np.array(dss.Circuit.YNodeVArray())[1::2]

# per-unit base: line-to-neutral base volts per node
base_ln = {}
for b in dss.Circuit.AllBusNames():
    dss.Circuit.SetActiveBus(b)
    kvb = dss.Bus.kVBase()          # already line-to-neutral kV
    for nd in dss.Bus.Nodes():
        base_ln[f"{b}.{nd}".upper()] = kvb*1000.0
Vbase = np.array([base_ln[o.upper()] for o in order])
Vpu = Vc / Vbase

# Ybus (siemens) -> per-unit on a common S base
Yd = np.array(dss.Circuit.SystemY())
Y = (Yd[0::2] + 1j*Yd[1::2]).reshape(N, N)
Sbase = 1e6                                   # 1 MVA
Zb = (Vbase[:, None]*Vbase[None, :])/Sbase    # elementwise base impedance
Ypu = Y * (Zb/1.0)                            # Y_pu = Y * Vb^2/Sbase (elementwise ok for VbxVb)

# ---- merge zero-impedance-connected nodes (switches / ideal regulators) ----
# Off-diagonal |Ypu| is ~50-150 for real branches and ~5.8e6 for zero-impedance
# links, so 1e4 separates them cleanly. Nodes joined by such links have identical
# voltages and are not independently observable; standard DSSE merges them.
ZI_THRESH = 1e4
_off = np.abs(Ypu - np.diag(np.diag(Ypu)))
_par = list(range(N))
def _find(a):
    while _par[a] != a:
        _par[a] = _par[_par[a]]; a = _par[a]
    return a
for _a, _b in zip(*np.where(_off > ZI_THRESH)):
    ra, rb = _find(int(_a)), _find(int(_b))
    if ra != rb: _par[ra] = rb
roots = sorted({_find(i) for i in range(N)})
sn_of = {r: k for k, r in enumerate(roots)}          # supernode index
node2sn = np.array([sn_of[_find(i)] for i in range(N)])
S = len(roots)
members = [[i for i in range(N) if node2sn[i] == k] for k in range(S)]
print(f"merged {N} nodes -> {S} supernodes (zero-impedance threshold {ZI_THRESH:g})")

slack_nodes = [i for i, o in enumerate(order) if o.upper().startswith("150.")][:3]
slack = sorted({int(node2sn[i]) for i in slack_nodes})       # slack SUPERNODES
theta_idx = [k for k in range(S) if k not in slack]
n_state = len(theta_idx) + S

Vm, Va = np.abs(Vpu), np.angle(Vpu)
G, B = Ypu.real, Ypu.imag

# ------------------------------------------------- analytic injection Jacobian
def injection_jacobian():
    """dP/dtheta, dP/dVm, dQ/dtheta, dQ/dVm for multiphase nodal injections."""
    th = Va[:, None] - Va[None, :]
    C, S = np.cos(th), np.sin(th)
    VV = Vm[:, None]*Vm[None, :]
    P = (VV*(G*C + B*S)).sum(1)
    Q_ = (VV*(G*S - B*C)).sum(1)
    dPdt = VV*(G*S - B*C); np.fill_diagonal(dPdt, 0.0)
    np.fill_diagonal(dPdt, -Q_ - B.diagonal()*Vm**2)
    dQdt = -VV*(G*C + B*S); np.fill_diagonal(dQdt, 0.0)
    np.fill_diagonal(dQdt, P - G.diagonal()*Vm**2)
    dPdV = Vm[:, None]*(G*C + B*S); np.fill_diagonal(dPdV, 0.0)
    np.fill_diagonal(dPdV, P/np.maximum(Vm, 1e-9) + G.diagonal()*Vm)
    dQdV = Vm[:, None]*(G*S - B*C); np.fill_diagonal(dQdV, 0.0)
    np.fill_diagonal(dQdV, Q_/np.maximum(Vm, 1e-9) - B.diagonal()*Vm)
    return dPdt, dPdV, dQdt, dQdV

dPdt, dPdV, dQdt, dQdV = injection_jacobian()

def _merge(vec_nodes):
    """sum node-level sensitivities into supernode columns (V is shared in a group)"""
    out = np.zeros(S)
    np.add.at(out, node2sn, vec_nodes)
    return out

def row_vmag(i):
    r = np.zeros(n_state); r[len(theta_idx) + int(node2sn[i])] = 1.0; return r

def row_inj(i, kind):
    r = np.zeros(n_state)
    dt, dv = (dPdt, dPdV) if kind == "P" else (dQdt, dQdV)
    r[:len(theta_idx)] = _merge(dt[i, :])[theta_idx]
    r[len(theta_idx):] = _merge(dv[i, :])
    return r

# ------------------------------------------------------- measurement design
load_nodes = set()
dss.Loads.First()
while True:
    nm = dss.CktElement.BusNames()[0].upper()
    bus = nm.split(".")[0]; nds = nm.split(".")[1:] or ["1", "2", "3"]
    for d in nds: load_nodes.add(f"{bus}.{d}")
    if dss.Loads.Next() == 0: break
idx_of = {o.upper(): i for i, o in enumerate(order)}
load_idx = sorted({idx_of[x] for x in load_nodes if x in idx_of})

# telemetry: substation 3-phase P,Q + V; regulator buses; a sparse set of remote V
reg_buses = ["150R", "9R", "25R", "160R"]
tele_v = [i for i, o in enumerate(order)
          if o.split(".")[0].upper() in ["150", "150R", "9R", "25R", "160R", "76", "300", "97", "67", "610"]]
tele_pq = [i for i, o in enumerate(order)
           if o.split(".")[0].upper() in ["76", "97", "300"] and i not in slack]

rows, sig, kind = [], [], []
for i in tele_v:
    rows.append(row_vmag(i)); sig.append(0.005); kind.append(f"Vmag@{order[i]}")     # 0.5% PT
for i in tele_pq:
    rows.append(row_inj(i, "P")); sig.append(0.01); kind.append(f"P@{order[i]}")
    rows.append(row_inj(i, "Q")); sig.append(0.01); kind.append(f"Q@{order[i]}")
H_tel = np.array(rows); s_tel = np.array(sig)**2

prow, psig, pkind = [], [], []
load_set = set(load_idx)
for i in range(N):
    if int(node2sn[i]) in slack:                      # never place injections at the stiff source
        continue
    if i in load_set:
        sg, tag = 0.50, "pseudoLoad"                  # load pseudo-measurement: 50% sigma
    else:
        sg, tag = 0.01, "zeroInj"                     # zero-injection: tight, standard DSSE
    prow.append(row_inj(i, "P")); psig.append(sg); pkind.append(f"{tag}P@{order[i]}")
    prow.append(row_inj(i, "Q")); psig.append(sg); pkind.append(f"{tag}Q@{order[i]}")
H_pse = np.array(prow); s_pse = np.array(psig)**2

# -------------------------------------------------------- empirical Q (G2)
print("estimating Q from load-varying sequence ...")
states = []
dss.Text.Command("New Loadshape.dummy npts=1 interval=1 mult=(1.0)")
for k in range(120):                                   # 120 x 1 s snapshots
    mult = 1.0 + 0.02*rng.standard_normal()            # aggregate load jitter
    dss.Text.Command(f"Set LoadMult={mult:.5f}")
    dss.Solution.Solve()
    v = np.array(dss.Circuit.YNodeVArray())[0::2] + 1j*np.array(dss.Circuit.YNodeVArray())[1::2]
    vpu = v/Vbase
    th_sn = np.array([np.angle(vpu)[members[k][0]] for k in range(S)])
    vm_sn = np.array([np.abs(vpu)[members[k][0]] for k in range(S)])
    states.append(np.r_[th_sn[theta_idx], vm_sn])
dss.Text.Command("Set LoadMult=1.0"); dss.Solution.Solve()
X = np.array(states); dX = np.diff(X, axis=0)
Qfull = np.cov(dX.T)/1.0
Qfull += np.eye(n_state)*1e-12                          # numerical floor
print(f"  Q: n={n_state}  trace={np.trace(Qfull):.3e}  max diag={np.max(np.diag(Qfull)):.3e}")

# ------------------------------------------------------------------ export
def rank_report(H, s2, tag):
    """lambda(G) = singular_values(A)^2 with A = H/sigma  -- guaranteed non-negative."""
    A = H/np.sqrt(s2)[:, None]
    sv = np.linalg.svd(A, compute_uv=False)
    lam = sv**2
    lmin = lam[-1] if len(lam) >= n_state else 0.0     # rank-deficient if m < n
    rank = int((sv > sv[0]*1e-10).sum())
    print(f"  {tag:26s} m={len(H):4d}  lam_min={lmin:.4g}  lam_max={lam[0]:.4g}  "
          f"kappa={lam[0]/lmin if lmin>0 else float('inf'):.3g}  rank={rank}/{n_state}")
    return lmin

print("\nobservability check:")
lm_t = rank_report(H_tel, s_tel, "telemetry only")
H_all = np.vstack([H_tel, H_pse]); s_all = np.r_[s_tel, s_pse]
lm_a = rank_report(H_all, s_all, "telemetry + pseudo")

np.savez_compressed(
    "feeder.npz",
    H=H_all, sigma2=s_all, Q=Qfull,
    H_telemetry=H_tel, sigma2_telemetry=s_tel,
    H_pseudo=H_pse, sigma2_pseudo=s_pse,
    n_telemetry=len(H_tel), node_order=np.array(order),
    meta=json.dumps(dict(
        source="IEEE 123-node test feeder, OpenDSS (dss-extensions/electricdss-tst)",
        engine=dss.Basic.Version(), nodes=N, states=n_state,
        Sbase_VA=Sbase, slack_nodes=[order[i] for i in slack],
        telemetry_channels=len(H_tel), pseudo_channels=len(H_pse),
        lam_min_telemetry_only=float(lm_t), lam_min_with_pseudo=float(lm_a),
        note="V-mag telemetry at bus 610 references the transformer-isolated 480V section (zero-sequence otherwise unobservable). H/sigma2 = telemetry+pseudo (what a real DSSE uses). "
             "H_telemetry alone is the starvable channel set.")))
print("\nwrote feeder.npz")
