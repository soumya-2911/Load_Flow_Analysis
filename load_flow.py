#!/usr/bin/env python3

#Need linedata.csv and busdata.csv
# linedata.csv

    # Column 0: From-bus number (integer)

    # Column 1: To-bus number (integer)

    # Column 2: Resistance R (per unit, float)

    # Column 3: Reactance X (per unit, float)

    # Column 4: Total line charging susceptance Bc (per unit, float)

    #| column 5* | Zero-sequence resistance R0 (per unit) optional | Float 

    # | column 6* | Zero-sequence reactance X0 (per unit) optional | Float 

    # | column 7* | Zero-sequence charging Bc0 (per unit) optional

# busdata.csv

    #   Column 0: Bus number (integer)

    #   Column 1: Bus type (integer: 0 = PQ, 1 = slack, 2 = PV)

    #   Column 2: Voltage magnitude Vm initial guess (per unit, float)

    #   Column 3: Voltage angle θ initial guess (degrees, float)

    #   Column 4: Active power demand Pd (MW, float)

    #   Column 5: Reactive power demand Qd (Mvar, float)

    #   Column 6: Active power generation Pg (MW, float)

    #   Column 7: Reactive power generation Qg (Mvar, float)

    #| 8 | Fault flag (0 = healthy, 1 = faulted) 

    # | 9 | Fault type (1 = SLG, 2 = LL, 3 = DLG, 4 = 3-ϕ) 

    # | 10 | Fault resistance Rf (per unit) 

    # | 11 | Fault reactance Xf (per unit) 


import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
import json
from typing import Tuple
import numpy as np
import pandas as pd

warnings.simplefilter("ignore", RuntimeWarning)          # Cleaner console


# ---------------------------------------------------------------------------
# Utility dataclasses
# ---------------------------------------------------------------------------
@dataclass
class SystemData:
    fb:  np.ndarray     # from-bus indices
    tb:  np.ndarray     # to-bus indices
    R:   np.ndarray     # branch resistance [p.u.]
    X:   np.ndarray     # branch reactance  [p.u.]
    Bc:  np.ndarray     # total line charging susceptance (B) [p.u.]
    R0: np.ndarray | None = None   # zero-seq R  (optional)
    X0: np.ndarray | None = None   # zero-seq X  (optional)
    Bc0: np.ndarray | None = None  # zero-seq Bc (optional)

    @property
    def Z0(self) -> np.ndarray:
        """Zero-sequence impedance per branch (falls back to positive)."""
        if self.R0 is None or self.X0 is None:
            return self.R + 1j * self.X
        return self.R0 + 1j * self.X0
    
    @property
    def Z(self) -> np.ndarray:
        """Series impedance per branch."""
       
        return self.R + 1j * self.X

    @property
    def y(self) -> np.ndarray:
        """Series admittance per branch (inverse of Z)."""
        return 1 / self.Z

    @property
    def nbus(self) -> int:
        return int(max(self.fb.max(), self.tb.max()))

    @property
    def nbranch(self) -> int:
        return len(self.fb)


@dataclass
class BusData:
    No:   np.ndarray
    Type: np.ndarray          # 0-PQ, 1-Slack, 2-PV
    Vm:   np.ndarray
    ang:  np.ndarray
    Pd:   np.ndarray
    Qd:   np.ndarray
    Pg:   np.ndarray
    Qg:   np.ndarray
    Fault:  np.ndarray   # 0/1
    Ftype:  np.ndarray   # 0-4
    Zf_R:   np.ndarray   # p.u.
    Zf_X:   np.ndarray

    @property
    def Zf(self):               # complex vector
        return self.Zf_R + 1j * self.Zf_X

    #Shorthand properties
    @property
    def nbus(self) -> int:
        return len(self.No)


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

# def read_excel_data(line_file: str = "linedata.csv",
#                     bus_file:  str = "busdata.csv") -> tuple[SystemData, BusData]:
#     """
#     Read line and bus data from Excel workbooks.
#     Sheet names do NOT matter – the first sheet is used.
#     Columns must match the original MATLAB order.
#     """

#     line_df = pd.read_csv(Path(line_file), header=None)
#     bus_df  = pd.read_csv(Path(bus_file),  header=None)
#     # Detect whether the sheet has 8 columns; if not, fall back
#     have_seq = line_df.shape[1] >= 8

#     sysdata = SystemData(
#         fb=line_df.iloc[1:, 0].astype(int).to_numpy(),
#         tb=line_df.iloc[1:, 1].astype(int).to_numpy(),
#         R=line_df.iloc[1:, 2].astype(float).to_numpy(),
#         X=line_df.iloc[1:, 3].astype(float).to_numpy(),
#         Bc=line_df.iloc[1:, 4].astype(float).to_numpy(),
#         R0 = line_df.iloc[1:,5].astype(float).to_numpy() if have_seq else None,
#         X0 = line_df.iloc[1:,6].astype(float).to_numpy() if have_seq else None,
#         Bc0= line_df.iloc[1:,7].astype(float).to_numpy() if have_seq else None
#     )

#     busdata = BusData(
#         No=bus_df.iloc[1:, 0].astype(int).to_numpy(),
#         Type=bus_df.iloc[1:, 1].astype(int).to_numpy(),
#         Vm=bus_df.iloc[1:, 2].astype(float).to_numpy(),
#         ang=bus_df.iloc[1:, 3].astype(float).to_numpy(),
#         Pd=bus_df.iloc[1:, 4].astype(float).to_numpy(),
#         Qd=bus_df.iloc[1:, 5].astype(float).to_numpy(),
#         Pg=bus_df.iloc[1:, 6].astype(float).to_numpy(),
#         Qg=bus_df.iloc[1:, 7].astype(float).to_numpy(),
#         Fault   = bus_df.iloc[1:, 8].fillna(0).to_numpy().astype(int),
#         Ftype   = bus_df.iloc[1:, 9].fillna(0).to_numpy().astype(int),
#         Zf_R    = bus_df.iloc[1:,10].fillna(0).astype(float).to_numpy(),
#         Zf_X    = bus_df.iloc[1:,11].fillna(0).astype(float).to_numpy()
#     )

#     return sysdata, busdata

def read_json_data(
    line_file: str = "linedata.json",
    bus_file: str = "busdata.json",time_stamp: int = 0
) -> Tuple[SystemData, BusData]:
    """
    Read line and bus data from separate JSON files.
    
    Parameters
    ----------
    line_file : str
        Path to JSON file containing line data
    bus_file : str  
        Path to JSON file containing bus data
        
    Returns
    -------
    (SystemData, BusData)
    """
    
    # Read line data from JSON file
    with open(line_file, 'r', encoding='utf-8') as f:
        line_data = json.load(f)
    
    # Read bus data from JSON file  
    with open(bus_file, 'r', encoding='utf-8') as f:
        bus_data = json.load(f)
    
    # Convert to numpy arrays (skip header row)
    line_rows = line_data[time_stamp][1:]  # Skip header row
    bus_rows = bus_data[time_stamp][1:]    # Skip header row
    
    # Check if we have sequence data (8+ columns)
    have_seq = len(line_data[0]) >= 8
    
    # Convert to numpy arrays
    line_array = np.asarray(line_rows, dtype=float)
    bus_array = np.asarray(bus_rows, dtype=float)
    
    # Build SystemData
    sysdata = SystemData(
        fb  = line_array[:, 0].astype(int),
        tb  = line_array[:, 1].astype(int), 
        R   = line_array[:, 2],
        X   = line_array[:, 3],
        Bc  = line_array[:, 4],
        R0  = line_array[:, 5] if have_seq else None,
        X0  = line_array[:, 6] if have_seq else None,
        Bc0 = line_array[:, 7] if have_seq else None,
    )
    
    # Build BusData - handle missing columns gracefully
    busdata = BusData(
        No    = bus_array[:, 0].astype(int),
        Type  = bus_array[:, 1].astype(int),
        Vm    = bus_array[:, 2],
        ang   = bus_array[:, 3], 
        Pd    = bus_array[:, 4],
        Qd    = bus_array[:, 5],
        Pg    = bus_array[:, 6],
        Qg    = bus_array[:, 7],
        Fault = bus_array[:, 8].astype(int) if bus_array.shape[1] > 8 else np.zeros(len(bus_rows), dtype=int),
        Ftype = bus_array[:, 9].astype(int) if bus_array.shape[1] > 9 else np.zeros(len(bus_rows), dtype=int),
        Zf_R  = bus_array[:,10] if bus_array.shape[1] > 10 else np.zeros(len(bus_rows)),
        Zf_X  = bus_array[:,11] if bus_array.shape[1] > 11 else np.zeros(len(bus_rows)),
    )
    
    return sysdata, busdata


# ---------------------------------------------------------------------------
# Y-bus constructor
# ---------------------------------------------------------------------------
def build_ybus(sysdata: SystemData) -> np.ndarray:
    """
    Symmetric nodal admittance matrix Ybus (nbus × nbus).

    Off-diagonals: –y_k
    Diagonals:     Σ y_k + j Bc/2
    """

    n = sysdata.nbus
    Y = np.zeros((n, n), dtype=complex)

    for k in range(sysdata.nbranch):
        y_series = sysdata.y[k]
        Bc_total = sysdata.Bc[k]
        i = sysdata.fb[k] - 1        # MATLAB is 1-based
        j = sysdata.tb[k] - 1
        # y = sysdata.y[k]

        Y[i, j] -= y_series
        Y[j, i] -= y_series

    diag_shunt = sysdata.y + 1j * sysdata.Bc / 2       # series + shunt
    for k in range(sysdata.nbranch):
        i = sysdata.fb[k] - 1
        j = sysdata.tb[k] - 1
        Y[i, i] += diag_shunt[k]
        Y[j, j] += diag_shunt[k]

    return Y

def build_ybus_sequence(sysdata: SystemData, sequence: str) -> np.ndarray:
    """
    Build a Y-bus for the requested symmetrical-components sequence.

    sequence: 'positive' | 'negative' | 'zero'
              (positive and negative are identical for most lines)
    """
    if sequence not in {"positive", "negative", "zero"}:
        raise ValueError("sequence must be 'positive', 'negative' or 'zero'")

    # Choose the impedance & charging arrays
    if sequence == "zero":
        Z = sysdata.Z0
        Bc = sysdata.Bc0 if sysdata.Bc0 is not None else sysdata.Bc
    else:                                   # positive or negative
        Z = sysdata.R + 1j * sysdata.X
        Bc = sysdata.Bc

    y = 1 / Z
    n = sysdata.nbus
    Y = np.zeros((n, n), dtype=complex)

    # Off-diagonals
    for k in range(sysdata.nbranch):
        i = sysdata.fb[k] - 1
        j = sysdata.tb[k] - 1
        Y[i, j] -= y[k]
        Y[j, i] -= y[k]

    # Diagonals
    diag = y + 1j * Bc / 2
    for k in range(sysdata.nbranch):
        i = sysdata.fb[k] - 1
        j = sysdata.tb[k] - 1
        Y[i, i] += diag[k]
        Y[j, j] += diag[k]

    return Y


# ---------------------------------------------------------------------------
# Newton-Raphson power flow
# ---------------------------------------------------------------------------
def newton_raphson(bus: BusData, Y: np.ndarray,
                   tol: float = 1e-4, maxiter: int = 80,
                   basemva: float = 100.0):
    """Full Newton-Raphson solver in polar form."""

    # Flat start if zero magnitude
    

    delta = np.deg2rad(bus.ang,dtype=float)
    Vm = np.where(bus.Vm <= 0, 1.0, bus.Vm)
    V = Vm * (np.cos(delta) + 1j * np.sin(delta))

    P = (bus.Pg - bus.Pd) / basemva
    Q = (bus.Qg - bus.Qd) / basemva

    slack = bus.Type == 1
    pv    = bus.Type == 2
    pq    = bus.Type == 0

    def mismatches(Vm, delta):
        V = Vm * (np.cos(delta) + 1j * np.sin(delta))
        S_calc = V * np.conj(Y @ V)
        Pm = P - S_calc.real
        Qm = Q - S_calc.imag
        return Pm, Qm, V

    for it in range(1, maxiter + 1):
        Pm, Qm, V = mismatches(Vm, delta)

        # Assemble Jacobian in sparse blocks
        n = bus.nbus
        G = Y.real
        B = Y.imag

        # Helper lambdas
        def dP_ddelta(i, j):
            if i == j:
                return -Q[i] - (Vm[i] ** 2) * B[i, i]
            return Vm[i] * Vm[j] * (G[i, j] * np.sin(delta[i] - delta[j]) -
                                     B[i, j] * np.cos(delta[i] - delta[j]))

        def dP_dV(i, j):
            if i == j:
                return P[i] / Vm[i] + G[i, i] * Vm[i]
            return Vm[i] * (G[i, j] * np.cos(delta[i] - delta[j]) +
                            B[i, j] * np.sin(delta[i] - delta[j]))

        def dQ_ddelta(i, j):
            if i == j:
                return P[i] - (Vm[i] ** 2) * G[i, i]
            return -Vm[i] * Vm[j] * (G[i, j] * np.cos(delta[i] - delta[j]) +
                                      B[i, j] * np.sin(delta[i] - delta[j]))

        def dQ_dV(i, j):
            if i == j:
                return Q[i] / Vm[i] - B[i, i] * Vm[i]
            return Vm[i] * (G[i, j] * np.sin(delta[i] - delta[j]) -
                            B[i, j] * np.cos(delta[i] - delta[j]))

        # Index maps
        pv_pq = np.where(~slack)[0]
        pq_only = np.where(pq)[0]
        mP = len(pv_pq)
        mQ = len(pq_only)

        # Jacobian sub-matrices
        J11 = np.zeros((mP, mP))
        J12 = np.zeros((mP, mQ))
        J21 = np.zeros((mQ, mP))
        J22 = np.zeros((mQ, mQ))

        for a, i in enumerate(pv_pq):
            for b, j in enumerate(pv_pq):
                J11[a, b] = dP_ddelta(i, j)

        for a, i in enumerate(pv_pq):
            for b, j in enumerate(pq_only):
                J12[a, b] = dP_dV(i, j)

        for a, i in enumerate(pq_only):
            for b, j in enumerate(pv_pq):
                J21[a, b] = dQ_ddelta(i, j)

        for a, i in enumerate(pq_only):
            for b, j in enumerate(pq_only):
                J22[a, b] = dQ_dV(i, j)

        # Compose and solve
        mismatch = np.hstack((Pm[pv_pq], Qm[pq_only]))
        J = np.block([[J11, J12],
                      [J21, J22]])

        DX = np.linalg.solve(J, mismatch)

        # Update
        delta[pv_pq] += DX[:mP]
        Vm[pq_only]  += DX[mP:]

        maxerr = abs(mismatch).max()
        if maxerr < tol:
            break
    else:
        print("WARNING: Newton-Raphson did not converge.")

    Vm_final = Vm
    ang_final = np.rad2deg(delta)
    return Vm_final, ang_final, it, V

# ---------------------------------------------------------------------------
# Line flow & loss calculation
# ---------------------------------------------------------------------------
def line_flows(sysdata: SystemData, V: np.ndarray,
               basemva: float = 100.0):
    fb = sysdata.fb - 1
    tb = sysdata.tb - 1
    y  = sysdata.y
    Bc = sysdata.Bc

    flows = []
    total_loss = 0 + 0j

    for k in range(sysdata.nbranch):
        i, j = fb[k], tb[k]

        Iij = (V[i] - V[j]) * y[k] + 1j * Bc[k] / 2 * V[i]
        Iji = (V[j] - V[i]) * y[k] + 1j * Bc[k] / 2 * V[j]

        Sij = V[i] * np.conj(Iij) * basemva
        Sji = V[j] * np.conj(Iji) * basemva
        loss = Sij + Sji

        flows.append((i + 1, j + 1, Sij.real, Sij.imag,
                      abs(Sij), loss.real, loss.imag))
        total_loss += loss

    return flows, total_loss / 2


# ---------------------------------------------------------------------------
# Pretty printers
# ---------------------------------------------------------------------------
def print_bus_table(bus: BusData, Vm: np.ndarray, ang: np.ndarray,
                    method: str, iters: int):
    head = (
        f"\nPower-flow solution – {method} ({iters} iterations)\n"
        " Bus |   |V|   |  Angle  |   Pd   |   Qd   |   Pg   |   Qg   |\n"
        "---------------------------------------------------------------"
    )
    print(head)
    for k in range(bus.nbus):
        print(f"{bus.No[k]:>4} | {Vm[k]:6.3f} | {ang[k]:8.3f} |"
              f"{bus.Pd[k]:7.2f} |{bus.Qd[k]:8.2f} |"
              f"{bus.Pg[k]:7.2f} |{bus.Qg[k]:8.2f} |")
    print("")


def print_line_flows(flows, total_loss):
    print("\nLine flows & losses:")
    print(" From  To |   P(MW)  Q(Mvar)  |  |S|  |  P-loss  Q-loss")
    print("-------------------------------------------------------")
    for (i, j, P, Q, S, Pl, Ql) in flows:
        print(f"{i:5}{j:4} | {P:8.2f}{Q:9.2f} | {S:6.2f} |"
              f"{Pl:8.2f}{Ql:9.2f}")
    print("-------------------------------------------------------")
    print(f"Total system loss: {total_loss.real:8.2f} MW, "
          f"{total_loss.imag:8.2f} Mvar\n")

#old
def ybus_with_fault(Y1,Y2,Y0,bus:BusData)->np.ndarray:
    """Return the post-fault positive-sequence Ybus seen by load-flow."""
    # If no fault flag is set, return Y1 untouched
    if bus.Fault.max()==0: return Y1

    k = int(np.where(bus.Fault==1)[0][0])        # first flagged bus
    ft = int(bus.Ftype[k]);  Zf = bus.Zf[k]
    
    
    if abs(Zf)<1e-9: Zf = 1e-9                  # solid fault → big adm

    if ft==4:    # 3-phase: only positive sequence affected
        Yfault = Y1.copy()
        Yfault[k,k] += 1/Zf
        return Yfault

    # Unsymmetrical faults – derive equivalent shunt admittance at k
    Z1 = np.linalg.inv(Y1)[k,k]
    Z2 = np.linalg.inv(Y2)[k,k]
    Z0 = np.linalg.inv(Y0)[k,k]

    # print("Z1",Z1)
    # print("Z2",Z2)
    # print("Z0",Z0)
    # print("Zf",Zf) #for testing

    if ft==1:    # SLG
        Z_eq = Z1 + Z2 + Z0 + 3*Zf
    elif ft==2:  # LL
        Z_eq = Z1 + Z2 + Zf
    elif ft==3:  # DLG
        
        Z_eq = ((Z1 + Zf)*(Z2 + Zf) + Z0*(Z1 + Z2 + Zf)) / (Z2 + Z0 + Zf)
    else:
        raise ValueError("Unknown fault type")

    # print("Zeq ",Z_eq )
    Y_shunt = 1/Z_eq              # shunt placed in positive-seq network
    Yfault = Y1.copy()
    # print("first",Yfault[k,k])

    # print("Yshunt ",Y_shunt)
    Yfault[k,k] += Y_shunt
    # print("second",Yfault[k,k])
    return Yfault



# ---------------------------------------------------------------------------
# Main menu driver
# ---------------------------------------------------------------------------
def main():
    sysdata, bus = read_json_data(time_stamp=0)            # edit paths if necessary
    # Ybus = build_ybus(sysdata)
    Y1 = build_ybus(sysdata)                       # positive sequence
    Y2 = Y1.copy()                                 # negative  (≈ Y1)
    Y0 = build_ybus_sequence(sysdata, 'zero')      # if R0/X0 provided


   

    while True:
        
        # Vm, ang, iters, V = newton_raphson(bus, Ybus)
        Yfault = ybus_with_fault(Y1,Y2,Y0,bus)     # may equal Y1
        
        Vm, ang, iters, V = newton_raphson(bus, Yfault,basemva=100)


        method = "Newton-Raphson"

        print_bus_table(bus, Vm, ang, method, iters)

        sub = input("Show Y-bus (y), line flows (l), or continue (Enter) or q (quit) ? ")
        if sub.lower().startswith("y"):
            print("\nFault-modified Ybus matrix (p.u.):\n", np.round(Yfault, 4))

        if sub.lower().startswith("q"):
            break

        if sub.lower().startswith("l"):
            flows, loss = line_flows(sysdata, V)
            print_line_flows(flows, loss)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nInterrupted by user")
