# Sounding_Rocket_VayuVega
This is the open simulator project for the model of sounding rocket done by me at IIMT UNIVERSITY, i welcome the changes if any. 
# 🚀 6-DOF Rocket Flight Simulator

## 📌 Project Overview

This project is a Python-based **6-Degree-of-Freedom (6-DOF) Rocket Flight Simulation Framework** designed to model the complete flight behavior of a rocket under realistic aerospace conditions.

Unlike simple altitude prediction models, this simulator considers the rocket as a rigid body and calculates its movement in three-dimensional space by combining:

- Translational motion
- Rotational motion
- Aerodynamic forces
- Propulsion
- Atmospheric effects
- Wind disturbances
- Structural vibration analysis

The objective of this project is to create an engineering simulation environment for studying rocket performance, stability, and structural behavior.

---

# ✨ Key Features

## 1. Six-Degree-of-Freedom Flight Dynamics

The simulation models all six degrees of freedom:

### Translational Motion:
- X-axis movement (Downrange)
- Y-axis movement (Lateral motion)
- Z-axis movement (Altitude)

### Rotational Motion:
- Roll
- Pitch
- Yaw

The equations of motion are solved using numerical integration through SciPy's Runge-Kutta solver.

---

# 🔥 Propulsion System Modeling

The simulator includes a rocket motor database containing:

- Motor total mass
- Propellant mass
- Burn duration
- Thrust-time curves

The thrust at any given time is calculated using interpolation of experimental thrust curves.

The model also accounts for:

- Propellant consumption
- Changing rocket mass
- Center of Gravity movement
- Variation in moment of inertia

---

# 🌎 Atmospheric and Wind Modeling

The simulator uses an International Standard Atmosphere (ISA) model to calculate:

- Air density
- Temperature variation
- Speed of sound

Different wind environments can be simulated:

- No wind
- Constant crosswind
- Wind shear
- Gusty wind conditions
- Custom wind profiles

Wind effects are transformed into the rocket body coordinate system for aerodynamic calculations.

---

# 🛰️ Aerodynamic Analysis

The aerodynamic model calculates:

- Drag force
- Normal force
- Side force
- Pitch moment
- Yaw moment
- Roll damping

The Center of Pressure (CP) is estimated using simplified Barrowman aerodynamic equations.

The simulator tracks the relationship between:

- Center of Gravity (CG)
- Center of Pressure (CP)

to evaluate rocket stability.

---

# ⚖️ Stability Analysis

During flight, the simulator calculates:

### Static Margin

\[
Static\ Margin = \frac{CP-CG}{Rocket\ Diameter}
\]

The static margin is monitored throughout the flight to determine whether the rocket remains stable during:

- Motor burn
- Maximum velocity
- Propellant depletion
- Descent phase

---

# 🏗️ Structural Vibration Analysis

The structural module estimates rocket bending frequencies using:

- Euler-Bernoulli beam theory
- Free-free beam vibration assumptions

It calculates:

- First bending mode frequency
- Second bending mode frequency

The simulator also checks possible resonance conditions caused by vortex shedding during high-speed flight.

---

# 📊 Simulation Outputs

The program generates:

### Rocket Geometry Visualization

Shows:
- Rocket body
- Nose cone
- Fins
- Center of Gravity
- Center of Pressure

### Flight Performance Graphs

Includes:

- Altitude vs Time
- Downrange Distance vs Time
- Velocity Profile

### Stability Graphs

Shows:

- Dynamic CG movement
- CP location
- Static Margin variation

### Structural Analysis Graphs

Displays:

- First bending mode frequency
- Second bending mode frequency

---

# 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| NumPy | Mathematical calculations |
| SciPy | Numerical integration |
| Matplotlib | Data visualization |
| ipywidgets | Interactive simulation controls |
| Object-Oriented Programming | Rocket system modelling |

---

# 📂 Project Structure




import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import ipywidgets as widgets
from IPython.display import display, clear_output

# =============================================================================
# 1. MOTOR DATABASE & AEROSPACE MATH CONTEXT
# =============================================================================

MOTOR_DATABASE = {
    "Aerotech G80 (Standard)": {
        "total_mass": 0.125,      # kg
        "propellant_mass": 0.062, # kg
        "burn_time": 1.40,        # s
        # Thrust Curve normalized time steps (0.0 to 1.0) mapped to Thrust (N)
        "thrust_curve": np.array(
            [[0.0, 0.0], [0.05, 95.0], [0.15, 82.0], [0.40, 80.0],
            [0.70, 78.0], [0.90, 72.0], [1.0, 0.0]]
        )
    },
    "Aerotech G64 (Long Burn)": {
        "total_mass": 0.118,
        "propellant_mass": 0.058,
        "burn_time": 1.90,
        "thrust_curve": np.array(
            [[0.0, 0.0], [0.05, 75.0], [0.20, 65.0], [0.50, 64.0],
            [0.80, 60.0], [0.95, 52.0], [1.0, 0.0]]
        )
    }
}

WIND_PROFILES = {
    "No Wind": lambda t, alt: (0.0, 0.0, 0.0), # (Wx, Wy, Wz)
    "Constant Crosswind (3 m/s)": lambda t, alt: (3.0, 0.0, 0.0),
    "Shear Wind (5 m/s at 100m)": lambda t, alt: (5.0 * min(alt / 100.0, 1.0), 0.0, 0.0),
    "Gusty Wind (Sinusoidal)": lambda t, alt: (3.0 + 2.0 * np.sin(0.5 * t), 0.0, 0.0)
}

def get_thrust(motor_name, t):
    motor = MOTOR_DATABASE[motor_name]
    if t >= motor["burn_time"]:
        return 0.0
    # Interpolate thrust curve
    norm_t = t / motor["burn_time"]
    times = motor["thrust_curve"][:, 0]
    thrusts = motor["thrust_curve"][:, 1]
    return float(np.interp(norm_t, times, thrusts))

def get_isa_atmosphere(altitude):
    """International Standard Atmosphere model up to troposphere layer."""
    T0 = 288.15; P0 = 101325.0; g0 = 9.80665; R = 287.05; L = 0.0065
    if altitude < 0: altitude = 0
    if altitude < 11000:
        T = T0 - L * altitude
        P = P0 * (T / T0) ** (g0 / (R * L))
    else:
        T = 216.65
        P = 22632.1 * np.exp(-g0 * (altitude - 11000) / (R * T))
    rho = P / (R * T)
    a = np.sqrt(1.4 * R * T)
    return rho, a

def quaternion_to_rotation_matrix(q):
    """Converts a unit quaternion [q0, q1, q2, q3] to a body-to-world rotation matrix."""
    q0, q1, q2, q3 = q / np.linalg.norm(q)
    return np.array([
        [1 - 2*(q2**2 + q3**2),   2*(q1*q2 - q0*q3),   2*(q1*q3 + q0*q2)],
        [2*(q1*q2 + q0*q3),   1 - 2*(q1**2 + q3**2),   2*(q2*q3 + q0*q1)],
        [2*(q1*q3 - q0*q2),   2*(q2*q3 + q0*q1),   1 - 2*(q1**2 + q2**2)]
    ])

# =============================================================================
# 2. CORE ROCKET RIGID BODY DEFINITION
# =============================================================================

class RocketVehicle:
    def __init__(self, inputs):
        self.struct_mass = inputs['struct_mass'] / 1000.0  # g to kg
        self.diameter = inputs['diameter'] / 1000.0        # mm to m
        self.length = inputs['length'] / 1000.0            # mm to m
        self.nose_length = inputs['nose_length'] / 1000.0  # mm to m
        self.fin_span = inputs['fin_span'] / 1000.0        # mm to m
        self.fin_chord = inputs['fin_chord'] / 1000.0      # mm to m
        self.num_fins = int(inputs['num_fins'])
        self.motor_name = inputs['motor_name']

        # Structural stiffness params
        self.E = inputs['youngs_modulus'] * 1e9            # GPa to Pa
        self.rho_material = inputs['material_density']     # kg/m^3

        # Static baseline Barrowman Calculations
        self.calculate_static_aerodynamics()

    def calculate_static_aerodynamics(self):
        # Nose Cone CP (Ogive/Conic estimation: 0.466 * length from tip)
        X_nose_cp = 0.466 * self.nose_length
        C_Na_nose = 2.0

        # Fins CP estimation (Simplified Barrowman)
        X_fins_leading_edge = self.length - self.fin_chord
        X_fins_cp = X_fins_leading_edge + (self.fin_chord / 3.0)
        # Standard fin lift coefficient slope approximation
        C_Na_fins = (self.num_fins * 2 * np.pi * (self.fin_span / self.diameter)**2) / (1 + np.sqrt(1 + (2 * self.fin_chord / self.fin_span)**2))

        # Total CP calculation from nose tip
        self.C_Na_total = C_Na_nose + C_Na_fins
        self.CP = ((C_Na_nose * X_nose_cp) + (C_Na_fins * X_fins_cp)) / self.C_Na_total
        self.ref_area = np.pi * (self.diameter / 2.0) ** 2

    def get_dynamic_mass_properties(self, t):
        """Calculates mass, CG location, and inertias accounting for motor burnout."""
        motor = MOTOR_DATABASE[self.motor_name]
        if t >= motor["burn_time"]:
            current_prop_mass = 0.0
        else:
            current_prop_mass = motor["propellant_mass"] * (1.0 - (t / motor["burn_time"])) * (1.0 - np.sqrt(t/motor["burn_time"]))
        current_motor_mass = (motor["total_mass"] - motor["propellant_mass"]) + current_prop_mass
        total_mass = self.struct_mass + current_motor_mass

        # Composite CG calculation (Assuming uniform structure + tail-heavy motor cluster)
        cg_struct = self.length * 0.45
        cg_motor = self.length - 0.05
        CG = ((self.struct_mass * cg_struct) + (current_motor_mass * cg_motor)) / total_mass

        # Moments of inertia calculations via Parallel Axis Theorem
        Ixx = 0.5 * total_mass * (self.diameter / 2.0)**2
        Iyy = (1.0/12.0) * total_mass * self.length**2 + total_mass * (CG - (self.length/2.0))**2
        Izz = Iyy

        # Time derivative of mass matrix (Coriolis matching fuel depletion)
        mdot = -motor["propellant_mass"] / motor["burn_time"] if t < motor["burn_time"] else 0.0

        return total_mass, CG, np.array([Ixx, Iyy, Izz]), mdot

# =============================================================================
# 3. 6-DOF RUNGE-KUTTA SIMULATION ENGINE
# =============================================================================

def simulation_6dof_kernel(t, state, vehicle, rail_length, launch_elevation, wind_profile_func):
    # State Vector Mapping:
    # state[0:3]   = Position World Frame (X, Y, Z)
    # state[3:6]   = Velocity Body Frame (u, v, w)
    # state[6:10]  = Quaternions (q0, q1, q2, q3)
    # state[10:13] = Angular Velocity Body Frame (p, q, r)

    pos = state[0:3]
    vel_b = state[3:6]
    q = state[6:10]
    q = q / np.linalg.norm(q) # Enforce unitary constraint
    omega = state[10:13]

    # Atmospheric properties
    rho, a_sound = get_isa_atmosphere(pos[2])
    V_mag_body = np.linalg.norm(vel_b)

    # Get wind vector from selected profile
    wind_w = np.array(wind_profile_func(t, pos[2])) # Pure horizontal crosswind

    # Transform Wind Vector into Body Frame
    R_b2w = quaternion_to_rotation_matrix(q)
    R_w2b = R_b2w.T
    wind_b = R_w2b @ wind_w

    # Effective airspeed calculation
    vel_rel_b = vel_b - wind_b
    V_air_mag = np.linalg.norm(vel_rel_b)

    # Dynamic mass updates
    mass, CG, I, mdot = vehicle.get_dynamic_mass_properties(t)

    # ------------------ FORCES & MOMENTS ------------------
    Thrust = get_thrust(vehicle.motor_name, t)
    F_thrust_b = np.array([0.0, 0.0, Thrust]) # Axial direction alignment

    # Basic Aerodynamic model computation
    F_aer_b = np.zeros(3)
    M_aer_b = np.zeros(3)

    if V_air_mag > 0.1:
        # Base Drag + Skin friction estimation
        Cd = 0.65 if V_air_mag / a_sound < 0.8 else 0.85
        q_dyn = 0.5 * rho * (V_air_mag**2)

        # Calculate localized angles of attack
        alpha = np.arctan2(vel_rel_b[1], vel_rel_b[2]) if vel_rel_b[2] != 0 else 0.0
        beta = np.arctan2(vel_rel_b[0], vel_rel_b[2]) if vel_rel_b[2] != 0 else 0.0

        # Normal & Side aerodynamic forces
        F_normal = q_dyn * vehicle.ref_area * vehicle.C_Na_total * alpha
        F_side = q_dyn * vehicle.ref_area * vehicle.C_Na_total * beta
        F_axial = q_dyn * vehicle.ref_area * Cd

        F_aer_b = np.array([-F_side, -F_normal, -F_axial])

        # Aerodynamic center offsets to evaluate restoration moments
        cp_moment_arm = vehicle.CP - CG
        M_aer_b[0] = -F_normal * cp_moment_arm - (0.1 * q_dyn * vehicle.ref_area * omega[0]) # Pitch damping
        M_aer_b[1] = F_side * cp_moment_arm - (0.1 * q_dyn * vehicle.ref_area * omega[1])   # Yaw damping
        M_aer_b[2] = -0.01 * q_dyn * vehicle.ref_area * omega[2]                            # Roll damping

    # Gravity computation mapping
    g_w = np.array([0.0, 0.0, -9.80665])
    g_b = R_w2b @ g_w
    F_gravity_b = g_b * mass

    # Total Structural System Matrices
    Total_F_b = F_thrust_b + F_aer_b + F_gravity_b

    # Check Launch Rail Constraints
    if np.linalg.norm(pos) < rail_length:
        # Constrain motion directly along the longitudinal axis vector
        Total_F_b[0] = 0.0
        Total_F_b[1] = 0.0
        if Total_F_b[2] < 0.0: Total_F_b[2] = 0.0
        M_aer_b = np.zeros(3)
        omega = np.zeros(3)

    # ------------------ STATE DERIVATIVES ------------------
    # Position derivative (Translating body frame velocities to world coordinates)
    dpos_dt = R_b2w @ vel_b

    # Rigid body translation accelerations (Euler equations)
    dvel_dt = (Total_F_b / mass) - np.cross(omega, vel_b)

    # Quaternion rate kinematics matrices
    dq_dt = 0.5 * np.array([
        -q[1]*omega[0] - q[2]*omega[1] - q[3]*omega[2],
         q[0]*omega[0] + q[2]*omega[2] - q[3]*omega[1],
         q[0]*omega[1] - q[1]*omega[2] + q[3]*omega[0],
         q[0]*omega[2] + q[1]*omega[1] - q[2]*omega[0]
    ])

    # Rigid body rotation dynamics (Euler rotational equations accounting for fuel Coriolis effects)
    domega_dt = np.zeros(3)
    domega_dt[0] = (M_aer_b[0] - (I[2] - I[1]) * omega[1] * omega[2]) / I[0]
    domega_dt[1] = (M_aer_b[1] - (I[0] - I[2]) * omega[0] * omega[2]) / I[1]
    domega_dt[2] = (M_aer_b[2] - (I[1] - I[0]) * omega[0] * omega[1]) / I[2]

    return np.concatenate([dpos_dt, dvel_dt, dq_dt, domega_dt])

# =============================================================================
# 4. STRUCTURAL VIBRATION DECK
# =============================================================================

def compute_structural_vibrations(vehicle, flight_time, velocity_profile, altitude_profile):
    """Models structural natural frequencies based on dynamic free-free beam formulation."""
    L = vehicle.length
    A = np.pi * (vehicle.diameter / 2.0)**2
    # Estimation of structural moment of inertia for a thin walled tube cylinder
    I_area = (np.pi / 64.0) * (vehicle.diameter**4 - (vehicle.diameter - 0.004)**4)

    freq_1st = []
    freq_2nd = []
    max_q_alert = False

    for t, v, alt in zip(flight_time, velocity_profile, altitude_profile):
        rho, _ = get_isa_atmosphere(alt)
        q_dyn = 0.5 * rho * (v**2)

        # Dynamic structural mass modification
        mass, _, _, _ = vehicle.get_dynamic_mass_properties(t)
        mu = mass / L # Linear density mass distribution representation

        # Free-free beam Euler-Bernoulli eigenvalue coefficient constants
        beta_1 = 4.73004 / L
        beta_2 = 7.8532 / L

        f1 = (beta_1**2 / (2 * np.pi)) * np.sqrt((vehicle.E * I_area) / mu)
        f2 = (beta_2**2 / (2 * np.pi)) * np.sqrt((vehicle.E * I_area) / mu)

        # Simple vortex shedding tracking computation
        St = 0.21 # Strouhal constant
        vortex_freq = (St * v) / vehicle.diameter if v > 1 else 0.0

        if np.abs(vortex_freq - f1) < 5.0 and q_dyn > 1000:
            max_q_alert = True

        freq_1st.append(f1)
        freq_2nd.append(f2)

    return np.array(freq_1st), np.array(freq_2nd), max_q_alert

# =============================================================================
# 5. HIGH RESOLUTION OPENROCKET-STYLE GUI CONTEXT
#    (Modified to use ipywidgets for Colab interactivity)
# =============================================================================

# The OpenRocket6DOFApp class is no longer used for interactive UI in Colab
# as it relies on tkinter, which is not supported.

# Refactoring the simulation and plotting logic to be called directly
def run_simulation_and_plot(inputs):
    # First, draw the schematic
    fig_schematic, ax_schematic = plt.subplots(figsize=(8, 4), facecolor='#1E1E1E')
    ax_schematic.set_facecolor('#2D2D2D')

    L_m = inputs['length'] / 1000.0
    D_m = inputs['diameter'] / 1000.0
    Ln_m = inputs['nose_length'] / 1000.0
    Fc_m = inputs['fin_chord'] / 1000.0
    Fs_m = inputs['fin_span'] / 1000.0

    ax_schematic.plot([0, Ln_m*1000, Ln_m*1000, 0], [0, D_m/2*1000, -D_m/2*1000, 0], color='#007ACC', lw=2)
    ax_schematic.plot([Ln_m*1000, L_m*1000, L_m*1000, Ln_m*1000], [D_m/2*1000, D_m/2*1000, -D_m/2*1000, -D_m/2*1000], color='white', lw=2)
    ax_schematic.plot([L_m*1000-Fc_m*1000, L_m*1000, L_m*1000, L_m*1000-Fc_m*1000], [D_m/2*1000, D_m/2*1000, D_m/2*1000+Fs_m*1000, D_m/2*1000], color='#E67E22', lw=2)
    ax_schematic.plot([L_m*1000-Fc_m*1000, L_m*1000, L_m*1000, L_m*1000-Fc_m*1000], [-D_m/2*1000, -D_m/2*1000, -(D_m/2*1000+Fs_m*1000), -D_m/2*1000], color='#E67E22', lw=2)

    # Instantiate vehicle physical systems for CP/CG calculation
    veh_temp = RocketVehicle(inputs)
    m_temp, cg_temp, _, _ = veh_temp.get_dynamic_mass_properties(0)
    ax_schematic.scatter([veh_temp.CP * 1000], [0], color='red', s=100, label=f'CP ({veh_temp.CP*1000:.1f}mm)', zorder=5)
    ax_schematic.scatter([cg_temp * 1000], [0], color='cyan', s=100, label=f'Initial CG ({cg_temp*1000:.1f}mm)', zorder=5)

    ax_schematic.set_title("Rocket Structural Geometry Validation Profiles", color='white')
    ax_schematic.tick_params(colors='white', labelcolor='white')
    ax_schematic.set_xlabel("Length (mm)", color='white')
    ax_schematic.set_ylabel("Radius (mm)", color='white')
    ax_schematic.legend(loc='upper right', facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    ax_schematic.axis('equal')
    plt.show()

    # Instantiate vehicle physical systems for simulation
    veh = RocketVehicle(inputs)

    theta = np.radians(90.0 - inputs['launch_elevation'])
    q0 = np.cos(theta / 2.0); q1 = 0.0; q2 = np.sin(theta / 2.0); q3 = 0.0

    init_state = np.array([
        0.0, 0.0, 0.1,  # Pos X, Y, Z
        0.0, 0.0, 0.0,  # Vel body u, v, w
        q0, q1, q2, q3, # Quaternions
        0.0, 0.0, 0.0   # Angular rate velocities p, q, r
    ])

    t_span = (0.0, 25.0)
    t_eval = np.linspace(0.0, 25.0, 600)

    def hit_ground_event(t, y, *args):
        return y[2] if t > 0.5 else 1.0
    hit_ground_event.terminal = True
    hit_ground_event.direction = -1

    # Get the selected wind profile function
    wind_profile_func = WIND_PROFILES[inputs['wind_profile_name']]

    sol = solve_ivp(
        simulation_6dof_kernel, t_span, init_state, t_eval=t_eval,
        args=(veh, inputs['rail_length'], inputs['launch_elevation'], wind_profile_func),
        events=hit_ground_event, method='RK45'
    )

    t = sol.t
    pos_x = sol.y[0]; pos_y = sol.y[1]; pos_z = sol.y[2]
    vel_b_z = sol.y[5]

    cg_history = []
    sm_history = []
    for ti in t:
        _, current_cg, _, _ = veh.get_dynamic_mass_properties(ti)
        cg_history.append(current_cg)
        sm_history.append((veh.CP - current_cg) / veh.diameter)

    f1, f2, alert = compute_structural_vibrations(veh, t, vel_b_z, pos_z)

    if alert:
        print("Structural Resonance Warning: Vortex shedding frequencies match the 1st bending natural frequency of the structure near Max-Q payload phases!")

    # Trajectory Plot
    fig1, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#1E1E1E')
    ax1.set_facecolor('#2D2D2D'); ax2.set_facecolor('#2D2D2D')

    ax1.plot(t, pos_z, 'g-', label='Altitude (Z)')
    ax1.plot(t, pos_x, 'b--', label='Downrange Flight (X)')
    ax1.set_title("Trajectory Space Optimization Data Arrays", color='white')
    ax1.tick_params(colors='white', labelcolor='white'); ax1.grid(True, alpha=0.3); ax1.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    ax1.set_ylabel("Distance (m)", color='white')

    ax2.plot(t, vel_b_z, 'r-', label='Axial Velocity (w)')
    ax2.set_xlabel("Time (seconds)", color='white')
    ax2.set_ylabel("Velocity (m/s)", color='white')
    ax2.tick_params(colors='white', labelcolor='white'); ax2.grid(True, alpha=0.3); ax2.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()

    # Stability Plot
    fig2, (ax3, ax4) = plt.subplots(2, 1, figsize=(10, 8), facecolor='#1E1E1E')
    ax3.set_facecolor('#2D2D2D'); ax4.set_facecolor('#2D2D2D')

    ax3.plot(t, np.array(cg_history)*1000, 'c-', label='Dynamic Center of Gravity (CG)')
    ax3.axhline(y=veh.CP*1000, color='r', linestyle='--', label='Center of Pressure (CP)')
    ax3.set_title("Static Stability Derivative Configurations", color='white')
    ax3.tick_params(colors='white', labelcolor='white'); ax3.grid(True, alpha=0.3); ax3.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    ax3.set_ylabel("Distance from nose (mm)", color='white')

    ax4.plot(t, sm_history, 'y-', label='Static Margin (Calibers)')
    ax4.axhline(y=1.5, color='gray', linestyle=':', label='Min Stable Margin (1.5 cal)')
    ax4.axhline(y=2.5, color='gray', linestyle=':', label='Max Stable Margin (2.5 cal)')
    ax4.set_xlabel("Time (seconds)", color='white')
    ax4.set_ylabel("Static Margin (Calibers)", color='white')
    ax4.tick_params(colors='white', labelcolor='white'); ax4.grid(True, alpha=0.3); ax4.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()

    # Vibration Plot
    fig3, ax5 = plt.subplots(figsize=(10, 6), facecolor='#1E1E1E')
    ax5.set_facecolor('#2D2D2D')

    ax5.plot(t, f1, 'm-', label='1st Bending Elastic Mode (Hz)')
    ax5.plot(t, f2, 'b-', label='2nd Bending Elastic Mode (Hz)')
    ax5.set_title("Euler-Bernoulli Elastic Free-Free Modal Frequencies", color='white')
    ax5.set_xlabel("Time (seconds)", color='white')
    ax5.set_ylabel("Frequency (Hz)", color='white')
    ax5.tick_params(colors='white', labelcolor='white'); ax5.grid(True, alpha=0.3); ax5.legend(facecolor='#2D2D2D', edgecolor='white', labelcolor='white')
    plt.tight_layout()
    plt.show()

# =============================================================================
# 6. EXECUTION CONTEXT (Modified for ipywidgets interactivity)
# =============================================================================

# --- Placeholder functions for loading external data ---
def load_motor_from_thrustcurve(motor_file_path, motor_name):
    # This is a placeholder function. You would need to implement
    # the actual parsing logic for .rse or .eng files.
    # Example: parsing a simplified .eng file format

    # For demonstration, let's assume a simple CSV-like format for thrust curve
    # and fixed values for other parameters.

    # In a real scenario, you'd read 'motor_file_path' and extract:
    # - total_mass (kg)
    # - propellant_mass (kg)
    # - burn_time (s)
    # - thrust_curve (np.array of [[time, thrust]])

    # Dummy data for illustration
    if "G75" in motor_name:
        # Example data if motor_file_path pointed to a G75 equivalent
        motor_data = {
            "total_mass": 0.150,
            "propellant_mass": 0.075,
            "burn_time": 1.6,
            "thrust_curve": np.array(
                [[0.0, 0.0], [0.05, 90.0], [0.20, 75.0], [0.60, 70.0],
                [0.90, 65.0], [1.0, 0.0]]
            )
        }
    else:
        print(f"Warning: No specific parsing logic for {motor_name}. Using generic data.")
        motor_data = {
            "total_mass": 0.100, # kg
            "propellant_mass": 0.050, # kg
            "burn_time": 1.2, # s
            "thrust_curve": np.array(
                [[0.0, 0.0], [0.1, 50.0], [0.5, 45.0], [0.9, 40.0], [1.0, 0.0]]
            )
        }

    MOTOR_DATABASE[motor_name] = motor_data
    print(f"Motor '{motor_name}' loaded into MOTOR_DATABASE.")

def fetch_and_create_wind_profile(location, timestamp, profile_name):
    # This is a placeholder function. You would need to implement
    # the actual API calls to Grosshopp or a weather service here.
    # The goal is to get wind data (speed, direction) across altitudes.

    print(f"Fetching wind data for {location} at {timestamp}...")

    # Dummy data for illustration based on location/timestamp
    # In a real scenario, this would involve API calls and data processing
    if "Cape Canaveral" in location:
        # Example: a more complex wind shear based on typical launch site conditions
        def cape_canaveral_wind(t, alt):
            # Simulate wind increasing with altitude and some time variation
            base_speed_x = 2.0 # m/s
            shear_factor = 0.05 # m/s per 100m
            max_alt_effect = 2000 # m

            wind_x = base_speed_x + min(alt, max_alt_effect) * (shear_factor / 100.0) # Wind shear

            # Add a small gust component based on time for variety
            gust_x = 1.0 * np.sin(t / 5.0) # Gust cycles every ~31 seconds

            return (wind_x + gust_x, 0.0, 0.0) # Assume only X-component for simplicity

        WIND_PROFILES[profile_name] = cape_canaveral_wind
        print(f"Wind profile '{profile_name}' created and added to WIND_PROFILES.")
    else:
        print(f"Warning: No specific wind data for {location}. Using default 'No Wind' profile.")
        WIND_PROFILES[profile_name] = WIND_PROFILES["No Wind"]


if __name__ == "__main__":
    # --- Demonstrate loading external data ---
    # Example of loading a motor from thrustcurve.org (placeholder implementation)
    # In a real scenario, 'path/to/Aerotech_G75.eng' would be a file you've uploaded or downloaded
    load_motor_from_thrustcurve('path/to/Aerotech_G75.eng', 'Aerotech G75 (Custom)') # Uncommented

    # Example of fetching and creating a wind profile (placeholder implementation)
    fetch_and_create_wind_profile("Cape Canaveral, FL", "2023-10-27T10:00:00Z", "Cape Canaveral October Wind") # Uncommented

    # Define widgets for interactive input
    struct_mass_widget = widgets.FloatSlider(
        value=600, min=100, max=1500, step=10, description='Structural Mass (g):'
    )
    length_widget = widgets.FloatSlider(
        value=900, min=300, max=2000, step=10, description='Rocket Length (mm):'
    )
    diameter_widget = widgets.FloatSlider(
        value=54, min=20, max=200, step=1, description='Fuselage Diameter (mm):'
    )
    nose_length_widget = widgets.FloatSlider(
        value=150, min=50, max=500, step=5, description='Nose Cone Length (mm):'
    )
    fin_chord_widget = widgets.FloatSlider(
        value=80, min=20, max=200, step=2, description='Fin Chord (mm):'
    )
    fin_span_widget = widgets.FloatSlider(
        value=50, min=10, max=150, step=1, description='Fin Span (mm):'
    )
    num_fins_widget = widgets.IntSlider(
        value=4, min=3, max=8, step=1, description='Number of Fins:'
    )
    youngs_modulus_widget = widgets.FloatSlider(
        value=70, min=10, max=200, step=1, description='Material Stiffness (GPa):'
    )
    material_density_widget = widgets.FloatSlider(
        value=2700, min=1000, max=8000, step=100, description='Material Density (kg/m3):'
    )
    rail_length_widget = widgets.FloatSlider(
        value=2.5, min=1.0, max=5.0, step=0.1, description='Launch Rail Length (m):'
    )
    launch_elevation_widget = widgets.FloatSlider(
        value=85, min=70, max=90, step=1, description='Launch Elevation (deg):'
    )

    # Update options for dropdowns after potential external data loading
    motor_options = list(MOTOR_DATABASE.keys())
    wind_options = list(WIND_PROFILES.keys())

    motor_name_widget = widgets.Dropdown(
        options=motor_options, value=motor_options[0], description='Select Motor:'
    )
    wind_profile_widget = widgets.Dropdown(
        options=wind_options, value=wind_options[0], description='Wind Profile:'
    )

    run_button = widgets.Button(description="RUN 6-DOF SIMULATION")
    output_area = widgets.Output()

    def on_run_button_clicked(b):
        with output_area:
            clear_output(wait=True)
            print("Running simulation...")
            current_inputs = {
                'struct_mass': struct_mass_widget.value,
                'length': length_widget.value,
                'diameter': diameter_widget.value,
                'nose_length': nose_length_widget.value,
                'fin_chord': fin_chord_widget.value,
                'fin_span': fin_span_widget.value,
                'num_fins': num_fins_widget.value,
                'youngs_modulus': youngs_modulus_widget.value,
                'material_density': material_density_widget.value,
                'rail_length': rail_length_widget.value,
                'launch_elevation': launch_elevation_widget.value,
                'wind_profile_name': wind_profile_widget.value,
                'motor_name': motor_name_widget.value
            }
            run_simulation_and_plot(current_inputs)
            print("Simulation complete.")

    run_button.on_click(on_run_button_clicked)

    # Display widgets
    display(widgets.VBox([
        struct_mass_widget, length_widget, diameter_widget, nose_length_widget,
        fin_chord_widget, fin_span_widget, num_fins_widget, youngs_modulus_widget,
        material_density_widget, rail_length_widget, launch_elevation_widget,
        wind_profile_widget, motor_name_widget, run_button, output_area
    ]))
