**# Load_Flow_Analysis**
Used Newton Raphson and Extended Kalman Filter to make a Real time Model of power system. Each of these are tested separately contained in the folder.

**Final Main File of State Estimator**
state_estimator.ipynb

**sag_train.json**

1) format: [dataset, 
            Angles, 
            List of Faulty Busses, 
            Fault_Type, 
            Scale]

2) dataset is the array of size num of train data each containing Z_k during a fault.

3) Format of each z_k is given below

4) Angles is the list of angle values . These angle values are itself list containing angles at each bus.

5) contains num of train data items. Each is a list informing in which busses the fault has occured. 0 implies bus number 1.

6) Fault Type is a list of num of train data points. Each is a string telling "sag" or "swell"

7) Scale is the list of num of train data points. Each is a integer informing by how much voltage values are scaled down or up during sag / swell.

8) Similar format for swell_train.json, normal_train.json


**bus_data.json**

1) Format: [Column Name,
            Data
            ]

2) Column_Name is an Array containing the Name of Columns

3) Data is an array containing each rows. If there is 14 busses, there are 14 rows


**line_data.json**

1) Format: [Column Name,
            Data
            ]

2) Column_Name is an Array containing the Name of Columns

3) Data is an array containing each rows. If there is 20 lines, there are 20 rows


**Bus labels**

1) Bus number
    A unique integer identifier for the bus (1–14).

2) Bus type

    0 = PQ (load) bus
    1 = Slack (swing) bus

    2 = PV (generator) bus



3) Voltage magnitude |V| (p.u.)
    initial guess of The per-unit voltage magnitude at that bus .

4) Voltage angle θ (degrees)
    initial Guess of The bus voltage phase angle in degrees (0 ° for the slack bus) .

5) P<sub>G> (p.u.)
    Real power generation injection at the bus (zero for pure loads).

6) Q<sub>G> (p.u.)
    Reactive power generation injection (zero for pure loads).

7) P<sub>L> (p.u.)
    Real power load demand at the bus (zero for pure generators).

8) Q<sub>L> (p.u.)
    Reactive power load demand at the bus.


**Line Labels**

1) From
2) To
3) resistance
4) reactance
5) susceptance
6) tap


**z_k**

14 bus voltage magnitudes (|V|)

14 real power injections (P)

14 reactive power injections (Q)

20 real power line flows (Pf)

20 reactive power line flows (Qf)
