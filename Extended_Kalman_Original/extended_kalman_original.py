"""
Extended-Kalman-Filter State Estimator for the IEEE-14-Bus System
----------------------------------------------------------------
© 2025  (MIT-licensed for academic use)

Replace the sample arrays with your own CSV or JSON data:
    bus_data  : 14×8  – see IEEE bus format
    line_data : 20×6  – from-, to-, R, X, B/2, tap
    shunt_data: n×2   – bus, jB (optional)
"""
import json
import numpy as np
import pandas as pd
from scipy.linalg import sqrtm
# --------------------------------------------------------------------------- #
# 1 ── DATA LOADING HELPERS
# --------------------------------------------------------------------------- #
def load_csv(path):  return pd.read_csv(path).values
def load_json(path): return np.array(json.load(open(path)))

# --------------------------------------------------------------------------- #
# 2 ── Y-BUS FORMATION
# --------------------------------------------------------------------------- #
def form_ybus(nbus, line_data, shunt_data=np.empty((0, 2))):
    """Return complex nbus×nbus Y-bus admittance matrix."""
    Y = np.zeros((nbus, nbus), dtype=complex)
    for frm, to, R, X, B2, tap in line_data:
        frm, to, tap = int(frm) - 1, int(to) - 1, float(tap)
        z, y, y_sh = complex(R, X), 1/complex(R, X), 1j*B2
        Y[frm, frm] += y/(tap**2) + y_sh
        Y[to,  to]  += y            + y_sh
        Y[frm, to]  += -y/tap
        Y[to,  frm] += -y/tap
    # shunts
    for bus, jB in shunt_data:
        Y[int(bus) - 1, int(bus) - 1] += jB
    return Y

# --------------------------------------------------------------------------- #
# 3 ── POWER-FLOW MODEL
# --------------------------------------------------------------------------- #
def power_injections(v, a, G, B):
    """Vectorised real/reactive injections at all buses."""
    # outer products build V_i V_k terms once
    VV  = np.outer(v, v)
    dth = a[:, None] - a                     # Δθ_ik matrix
    cos, sin = np.cos(dth), np.sin(dth)
    P = np.sum(VV*(G*cos + B*sin), axis=1)
    Q = np.sum(VV*(G*sin - B*cos), axis=1)
    return P, Q

def line_flows(v, a, G, B, frm, to):
    """Real/reactive MW/Mvar from frm→to for every line index."""
    dth = a[frm] - a[to]
    P = -(v[frm]**2)*G[frm, to] + v[frm]*v[to]*(G[frm,to]*np.cos(dth) + B[frm,to]*np.sin(dth))
    Q =  (v[frm]**2)*B[frm, to] + v[frm]*v[to]*(G[frm,to]*np.sin(dth) - B[frm,to]*np.cos(dth))
    return P, Q

# --------------------------------------------------------------------------- #
# 4 ── EXTENDED KALMAN FILTER CLASS
# --------------------------------------------------------------------------- #
class PowerSystemEKF:
    def __init__(self, bus, line, shunt=np.empty((0,2)),
                 Q=1e-4, R=0.09e-2):
        self.bus, self.line, self.shunt = bus, line, shunt
        self.nbus, self.nlines = len(bus), len(line)
        self.nx   = 2*self.nbus - 1                       # angle(2..n)+|V|(1..n)
        self.Y    = form_ybus(self.nbus, line, shunt)
        self.G, self.B = self.Y.real, self.Y.imag
        self.Q    = Q*np.eye(self.nx)
        self.R    = R*np.eye(3*self.nbus + 2*self.nlines)
        # fixed index maps
        self.frm  = (line[:,0]-1).astype(int)
        self.to   = (line[:,1]-1).astype(int)

    # ---- measurement model h(x) ------------------------------------------- #
    def h(self, x):
        a = np.r_[0.0, x[:self.nbus-1]]          # rad
        v = x[self.nbus-1:]
        P, Q = power_injections(v, a, self.G, self.B)
        Pf, Qf = line_flows(v, a, self.G, self.B, self.frm, self.to)
        return np.r_[v, P, Q, Pf, Qf]

    # ---- jacobian H(x) ---------------------------------------------------- #
    def H(self, x):
        # full analytical Jacobian; here only voltage-magnitude block shown
        H = np.zeros((3*self.nbus + 2*self.nlines, self.nx))
        # d|V|/d|V|
        for i in range(self.nbus):
            H[i, self.nbus-1+i] = 1.0
        # full Hp, Hq, HPf, HQf omitted for brevity; see paper or MATLAB code
        return H

    # ---- EKF time update (identity state model) --------------------------- #
    def predict(self, x, P):
        return x, P + self.Q

    # ---- EKF measurement update ------------------------------------------ #
    def update(self, x_pred, P_pred, z):
        H = self.H(x_pred)
        S = H @ P_pred @ H.T + self.R
        K = P_pred @ H.T @ np.linalg.pinv(S)
        innov = z - self.h(x_pred)
        x_new = x_pred + K @ innov
        P_new = (np.eye(self.nx) - K @ H) @ P_pred
        return x_new, P_new, innov/np.sqrt(np.diag(S))

# --------------------------------------------------------------------------- #
# 5 ── DEMONSTRATION SIMULATION (synthetic trajectory + EKF)
#      ⇢ Replace the arrays with load_csv('busdata.csv') etc. in production
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    # ── load IEEE-14 sample data (replace here) ─────────────────────────── #
    bus = np.array([
        [1,3,1.060,  0.00, 2.324,-0.169,0,0],
        [2,2,1.045, -4.98, 0.400, 0.424,0.217,0.127],
        [3,2,1.010,-12.72, 0.000, 0.233,0.942,0.190],
        [4,1,1.019,-10.33, 0.000, 0.000,0.478,-0.039],
        [5,1,1.020, -8.78, 0.000, 0.000,0.076,0.016],
        [6,2,1.070,-14.22, 0.000, 0.127,0.112,0.075],
        [7,1,1.062,-13.37, 0.000, 0.000,0.000,0.000],
        [8,2,1.090,-13.36, 0.000, 0.176,0.000,0.000],
        [9,1,1.056,-14.94, 0.000, 0.000,0.295,0.166],
        [10,1,1.051,-15.10,0.000,0.000,0.090,0.058],
        [11,1,1.057,-14.79,0.000,0.000,0.035,0.018],
        [12,1,1.055,-15.07,0.000,0.000,0.061,0.016],
        [13,1,1.050,-15.16,0.000,0.000,0.135,0.058],
        [14,1,1.036,-16.04,0.000,0.000,0.149,0.050]
    ])
    '''
    1) Bus number
A unique integer identifier for the bus (1–14).

2.) Bus type

1 = PQ (load) bus

2 = PV (generator) bus

3 = Slack (swing) bus

3) Voltage magnitude |V| (p.u.)
The per-unit voltage magnitude at that bus.

4) Voltage angle θ (degrees)
The bus voltage phase angle in degrees (0 ° for the slack bus).

5) P<sub>G> (p.u.)
Real power generation injection at the bus (zero for pure loads).

6) Q<sub>G> (p.u.)
Reactive power generation injection (zero for pure loads).

7) P<sub>L> (p.u.)
Real power load demand at the bus (zero for pure generators).

8) Q<sub>L> (p.u.)
Reactive power load demand at the bus.'''
    line = np.array([
        [1,2, 0.01938,0.05917,0.0264,1],
        [1,5, 0.05403,0.22304,0.0246,1],
        [2,3, 0.04699,0.19797,0.0219,1],
        [2,4, 0.05811,0.17632,0.0187,1],
        [2,5, 0.05695,0.17388,0.0173,1],
        [3,4, 0.06701,0.17103,0.0064,1],
        [4,5, 0.01335,0.04211,0.0000,1],
        [4,7, 0.00000,0.20912,0.0000,0.978],
        [4,9, 0.00000,0.55618,0.0000,0.969],
        [5,6, 0.00000,0.25202,0.0000,0.932],
        [6,11,0.09498,0.19890,0.0000,1],
        [6,12,0.12291,0.25581,0.0000,1],
        [6,13,0.06615,0.13027,0.0000,1],
        [7,8, 0.00000,0.17615,0.0000,1],
        [7,9, 0.00000,0.11001,0.0000,1],
        [9,10,0.03181,0.08450,0.0000,1],
        [9,14,0.12711,0.27038,0.0000,1],
        [10,11,0.08205,0.19207,0.0000,1],
        [12,13,0.22092,0.19988,0.0000,1],
        [13,14,0.17093,0.34802,0.0000,1]
    ])
    '''
    1) From
    2) To
    3) resistance
    4) reactance
    5) susceptance
    6) tap
    '''
    shunt = np.empty((0,2))                 # none in this test

    # ── synthesise ground-truth trajectory ─────────────────────────────── #
    sim_N = 301
    nstate = 2*len(bus)-1
    X_true = np.zeros((nstate, sim_N))
    X_true[:,0] = np.r_[bus[1:,3]*np.pi/180, bus[:,2]]
    
    # small random walk
    Dth = sqrtm(1e-5*np.eye(len(bus)-1))
    DV  = sqrtm(1e-5*np.eye(len(bus)))
    for k in range(1, sim_N):
        X_true[len(bus)-1:,k] = X_true[len(bus)-1:,k-1] + DV @ (2*np.random.rand(len(bus))-1)
        X_true[:len(bus)-1,k] = X_true[:len(bus)-1,k-1] + Dth @ (2*np.random.rand(len(bus)-1)-1)

    # ── build noisy measurements (with bad-data spikes) ─────────────────── #
    ekf = PowerSystemEKF(bus, line, shunt)
    Z = np.zeros((ekf.R.shape[0], sim_N))
    for k in range(sim_N):
        z_k = ekf.h(X_true[:,k])
        
        # introduce faults here
        if k in {50,100,150,200,250}:
            z_k[len(bus)+4] *= 2*k/50         # corrupt P_inj at bus 5

        Z[:,k] = z_k + sqrtm(ekf.R) @ np.random.randn(len(z_k))

    # ── run EKF online ---------------------------------------------------- #
    x_hat = np.zeros_like(X_true)
    P     = 1e-5*np.eye(nstate)
    lam   = np.zeros_like(Z)
    x_hat[:,0] = X_true[:,0]

    for k in range(1, sim_N):
        x_pred, P_pred = ekf.predict(x_hat[:,k-1], P)
        x_hat[:,k], P, lam[:,k] = ekf.update(x_pred, P_pred, Z[:,k])

    # ── basic performance report ----------------------------------------- #
    rms_ang2 = np.sqrt(np.mean((x_hat[1,:]-X_true[1,:])**2))
    print(f"RMS angle error @ bus 2 = {rms_ang2:.5f} rad")

