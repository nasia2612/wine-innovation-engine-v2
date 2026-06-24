# because we know the rate of change we solve the system of equations using solve_ivp
import numpy as np
from scipy.integrate import solve_ivp

import matplotlib.pyplot as plt

# --------------------------------PHASE 1:Process digital twin: ODE simulator -----------------------------------------------

# state vector :the numbers that we want to track during the simulation
STATE_NAMES = ["X", "Xt", "S", "N", "E"]
#   X  = viable yeast biomass (g/L)
#   Xt = total yeast biomass  (g/L)
#   S  = sugar (g/L)                  ← glucose + fructose
#   N  = assimilable nitrogen (mg/L)  ← if it finishes early→ stuck
#   E  = ethanol (g/L)                ← product + inhibitor

#'The ethanol yield is not 0.51 but ~0.47-0.49 because carbon is shared between ethanol, biomass and glycerol

# Parameters (white wine preset, literature values) ─────
params = {
    # Growth
    "mu_max": 0.05,  # — ~14°C (Nelson & Boulton, 2024)
    "KN": 8.8,  # half-saturation for nitrogen (mg/L) — Monod KS
    "kE": 0.035,  # ethanol inhibition constant — Coleman et al.
    "kd_prime": 0.000065,  # death rate (1/h) — viability decay
    # Stoichiometry
    "Yxn": 0.04,  # nitrogen consumed per biomass (4 g/L X from 100 mg/L N)
    "Yes": 0.47,  # ethanol yield on sugar (g E / g S) — Gay-Lussac ~0.51 ,coleman ~0.55
    # Sugar uptake
    "Ks": 2.0,  # half-saturation for sugar (g/L)
    "vmax_s": 0.10,  # max sugar uptake rate (g S / g X / h)
    # Temperature
    "T_ref": 15.0,  # reference temperature (°C) arrhenius
    # Initial conditions (white wine)
    "X0": 0.25,  # initial biomass (g/L)
    "Xt0": 0.25,
    "S0": 220.0,  # initial sugar (g/L) —
    "N0": 300.0,  # initial YAN (mg/L) — adequate nitrogen
    "E0": 0.0,  # no ethanol at start
    "T": 14.0,  # fermentation temperature (°C)
    "MNT": 0.05,  # g S / g X / h — simplified constant maintenance term for sugar consumption(the sugar consumption that is not directly related to growth, but rather to the maintenance of cellular functions. This term accounts for the energy and resources that the yeast cells need to maintain their viability and metabolic activity even when they are not actively growing. The maintenance term is important for long fermentations where the yeast may stop growing but still consume sugar for maintenance.)
}


# rhs() takes the current state and parameters and returns the 5 derivatives (dX/dt, dXt/dt, dS/dt, dN/dt, dE/dt) as a list or array.


def rhs(t, y, p):  # t as time,y the state vector,p as the parameters
    X, Xt, S, N, E = y  # unpack the state vector

    # Guard: concentrations cannot go below zero
    X = max(X, 0.0)
    S = max(S, 0.0)
    N = max(N, 0.0)
    E = max(E, 0.0)
    # modulating factors : nitrogen limitation, ethanol inhibition, temperature effect
    # AS F_N increases, growth rate increases, as F_N decreases, growth rate decreases

    # 3. Temperature
    R_gas = 8.314
    Ea = 55_000
    T_K = p["T"] + 273.15
    T_ref_K = p["T_ref"] + 273.15
    f_T = np.exp(-Ea / R_gas * (1 / T_K - 1 / T_ref_K))

    # nitrogen limitation (monod type)    , N=nitrogen concentration, KN=half-saturation constant for nitrogen
    f_N = N / (
        p["KN"] + N
    )  # as N increases, f_N approaches 1, as N decreases, f_N approaches 0

    # 2. Ethanol inhibition — όσο ανεβαίνει το E, πέφτει η δράση
    f_E = np.exp(-p["kE"] * E)
    #   # RATES

    # specific growth rate  -how fast the biomass grows per unit of existing biomass
    mu = p["mu_max"] * f_N * f_E * f_T  # modulated by the three factors

    # sugar uptake rate per biomass
    # wwe model the sugar uptake as a Monod-type function of sugar concentration (S) modulated by ethanol and temperature, but not directly by nitrogen, because nitrogen affects growth more than uptake.

    v_s = (
        p["vmax_s"] * (S / (p["Ks"] + S)) * f_T
    )  # modulated by ethanol and temperature

    # dervatives of the state variables
    kd = p["kd_prime"] * E
    # because i have a very small consant,i will try to make it ethanol-dependent, so that the death rate increases with ethanol concentration. This is a simple way to model the inhibitory effect of ethanol on yeast viability. The idea is that as ethanol accumulates in the medium, it becomes more toxic to the yeast cells, leading to an increased death rate. This modification can help capture the dynamics of yeast population decline in high-ethanol environments, which is relevant for wine fermentation where ethanol levels can become inhibitory.
    dX = mu * X - kd * X
    dXt = mu * X
    # we will put sugar gate to stop the yeast production when sugar is depleted, because we know that yeast will stop growing and producing ethanol when there is no sugar left. This is a simple way to model the end of fermentation, where the yeast cells enter a stationary phase due to nutrient limitation. The sugar gate ensures that the model reflects this biological reality, preventing unrealistic behavior such as negative sugar concentrations or continued growth in the absence of substrate.
    sugar_gate = S / (p["Ks"] + S)

    dS = (
        -v_s * X - p["MNT"] * X * sugar_gate
    )  # the main sugar consumption is due to the yeast growth, but there is also a maintenance term that consumes sugar even when the yeast is not growing. This maintenance term is proportional to the biomass and has a rate constant MNT (maintenance rate). The maintenance term is important for long fermentations where the yeast may stop growing but still consume sugar for maintenance.
    dN = -(mu * X) / p["Yxn"]  # was -p["Yxn"] * mu * X
    dE = (
        p["Yes"] * (v_s + p["MNT"] * sugar_gate) * X
    )  # anaerobic glycolisis->ethanol+co2

    return [dX, dXt, dS, dN, dE]


# so i will solve the system of equations using solve_ivp, which is a numerical integrator for ordinary differential equations (ODEs). This will allow us to simulate the fermentation process over time and observe how the state variables change.


def run_simulation(p=params, t_end=700, use_event=True):
    y0 = [p["X0"], p["Xt0"], p["S0"], p["N0"], p["E0"]]

    kwargs = dict(
        fun=rhs,
        t_span=(0, t_end),
        y0=y0,
        args=(p,),
        method="RK45",
        dense_output=True,
        max_step=1.0,
    )
    # we use event to find the when the function is zero
    # dense output because we want to fix our own grid
    if use_event:

        def sugar_depleted(t, y, p):
            return y[2]

        sugar_depleted.terminal = True
        sugar_depleted.direction = -1  # when it goes negative stop
        kwargs["events"] = sugar_depleted

    sol = solve_ivp(**kwargs)
    return sol


def plot_simulation(sol):
    t = sol.t
    X, Xt, S, N, E = sol.y

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("White Wine Fermentation — Digital Twin", fontsize=14)

    axes[0, 0].plot(t, X, color="green")
    axes[0, 0].set_title("Viable Biomass X (g/L)")
    axes[0, 1].plot(t, S, color="orange")
    axes[0, 1].set_title("Sugar S (g/L)")
    axes[0, 2].plot(t, E, color="purple")
    axes[0, 2].set_title("Ethanol E (g/L)")
    axes[1, 0].plot(t, N, color="blue")
    axes[1, 0].set_title("Nitrogen N (mg/L)")
    axes[1, 1].plot(t, Xt, color="gray")
    axes[1, 1].set_title("Total Biomass Xt (g/L)")

    for ax in axes.flat:
        ax.set_xlabel("Time (h)")
        ax.grid(True, alpha=0.3)

    axes[1, 2].axis("off")
    plt.tight_layout()
    plt.savefig(
        "results/figures/fermentation_simulation_kd_prime_dependant_1.png",
        dpi=150,
    )
    plt.show()


if __name__ == "__main__":
    sol = run_simulation()
    plot_simulation(sol)
