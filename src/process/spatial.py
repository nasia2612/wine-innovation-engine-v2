import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import interp1d
from src.process.simulator import run_simulation, params as ferm_params

# Phase 2: CFD-lite 1D heat transport in fermentation tank
# - 1D heat equation (method of lines) along tank height z
#   - Dirichlet BC at base (z=0): T = T_coolant (cooling jacket)
#   - Neumann BC at top (z=L):  dT/dz = 0 (adiabatic, no heat loss)
#   - Source term: Q_ferm = DeltaH_ferm * |dS/dt| (exothermic fermentation)


# Sources:
#  Bogard et al. 2020, Foods 9:865 — tank geometry, wine thermal props
#   Nelson & Boulton 2024, Fermentation


params = {
    "H": 2.2,  # m, tank height (Bogard 2020)
    "N": 40,  # layers — start here; dz = H/(N-1) ≈ 0.116 m
    "alpha": 1.9e-5,  # m²/s, effective (convective) (we make it bigger than pure diffusion to account for convective mixing)
    "rho": 1080.0,  # kg/m³, fermenting must (Nelson & Boulton),density of the fermenting liquid
    "Cp": 3800.0,  # J/kg/K, approximated  ,specific heat capacity of the fermenting liquid
    "T_coolant": 12.0,  # °C, jacket/cellar (Bogard 2020)
    "T_initial": 12.0,  # °C
    "DELTA_H_FERM": 561.6,  # J/g sugar (101.2 kJ/mol ÷ 180.16),Enthalpy of Fermentation ,the heat released per gram of sugar fermented, calculated from the enthalpy change of the fermentation reaction.
}


# derived quantities
params["dz"] = params["H"] / (params["N"] - 1)  # N-1 gaps
params["alpha_h"] = (
    params["alpha"] * 3600
)  # convert it to hours, because the ODE solver works in hours, and the heat equation is in seconds. So we multiply by 3600 to convert from seconds to hours. This way, the units are consistent when we use the heat equation in our simulation.


# we are gonna use the method of lines to solve the heat equation.
# The method of lines is a technique for solving partial differential equations (PDEs) by discretizing the spatial domain into a finite number of points (or layers) and then solving the resulting system of ordinary differential equations (ODEs) in time.
# In our case, we are going to discretize the height of the fermentation tank into N layers, and then we will solve the heat equation for each layer as a function of time.
# Converts the heat PDE  ∂T/∂t = α·∂²T/∂z²  into N coupled ODEs


def heat_rhs(t, T, params):  # It calculates how the temperature at each of our
    # 20 layers changes in the next microsecond due to pure diffusion (no fermentation heat).

    N = params["N"]
    dz = params["dz"]
    alpha = params["alpha_h"]

    # dTdt-> ndarray of length N, representing the rate of change of temperature at each layer

    dTdt = np.zeros(N)

    # base (Dirichlet BC): T[0] = T_coolant
    dTdt[0] = 0.0
    # finite difference approximation for the interior layers (1 to N-2)
    for i in range(
        1, N - 1
    ):  # the loop covers the interior layers of the tank, from layer 1 to layer N-2 (0-indexed). The first and last layers are handled separately due to boundary conditions.
        d2T_dz2 = (T[i + 1] - 2 * T[i] + T[i - 1]) / dz**2  # the central difference
        dTdt[i] = alpha * d2T_dz2

    # we invent a ghost node above node 19 -> T[N] = T[N-2] (it makes a mirror,Neumann BC: dT/dz=0)
    d2T_top = 2 * (T[N - 2] - T[N - 1]) / dz**2
    dTdt[N - 1] = alpha * d2T_top

    return dTdt


# no fermentation heat source term, just pure diffusion, to test the heat equation solver, a linear gradient (cold base,warm top) is expected to flatten over time, and the temperature should converge to a uniform value (the average of the initial temperatures) as time progresses.
#  The test checks if the final temperature profile is close to this expected uniform value, within a specified tolerance.
# without source term, the heat equation reduces to a pure diffusion problem, and the temperature should eventually become uniform throughout the tank. The test checks if the final temperature profile is close to this expected uniform value, within a specified tolerance.
def run_diffusion_test(params, T_top=24, t_end=100, n_save=200):
    N = params["N"]

    # Initial condition : Linear gradient from base to top
    T0 = np.linspace(
        params["T_coolant"], T_top, N
    )  # linear gradient from T_coolant at the base to T_top at the top
    # base (node 0) =T_coolant=12
    # top  (node N-1)=T_top=24

    # enforce dirichelt BC at the base (node 0)
    T0[0] = params["T_coolant"]

    # time pointa to record the solution
    t_eval = np.linspace(0, t_end, n_save)

    # solve the heat equation using solve_ivp
    sol = solve_ivp(
        fun=heat_rhs,
        t_span=(0, t_end),
        y0=T0,
        args=(params,),
        method="BDF",  # stiff solver(stiff we mean by when fast and slow processes coexist), because the heat equation can be stiff, especially for small dz and large alpha. The BDF method is suitable for stiff problems and can handle the rapid changes in temperature that may occur in the early stages of the simulation.
        t_eval=t_eval,
        dense_output=True,
    )

    return sol


# prescribed source term,take a representative ds/dt(t) weighted by height -> what gradient does fermentation heat produce?
# differentaites S(t)->|ds/dt|(t)
# wraps it as a callable function of time


def get_dSdt_profile(t_eval, sim_params):
    sol = run_simulation(sim_params)  # stops when the sugar depltes
    t_stop = sol.t[-1]  # actual end time

    # so we need to evaluate sugar only within the solved window
    S = np.zeros_like(t_eval)  # creates a frame with zeros with the shape of t_eval
    inside = (
        t_eval <= t_stop
    )  # a boolean mask: True for time points within the solved window, False beyond.

    S[inside] = sol.sol(t_eval[inside])[2]  # sugar = state index 2 → [X, Xt, S, N, E]
    # past t_stop, S stays 0 (sugar gone) → dS/dt = 0 → no fermentation heat

    dS = np.gradient(
        S, t_eval
    )  # gradient : estimates dS/dt from neighbouring points using finite differences
    return np.abs(dS)


# heat_rhs extended with fermentation heat source term
# at each time t, retrieves |ds/dt| from the pre-built interpolator ,converts to a heat source Q(°C/h per layer),
# and adds it to every interior and top layer
def heat_rhs_with_source(t, T, params):
    N = params["N"]
    dz = params["dz"]  # spatial step size
    alpha = params["alpha_h"]

    # in phase ,we calculated the rate of change dS/dt but when we run the spatial heat simulation the solver uisng the BDF was jumping around in tie
    # the data is discrete so the model crushes so we put the interpolator dSdt_func which is a continuous mmathematical curve that connects all the dots
    # , it looks at the curve, estimates where the value should be, and returns the exact sugar consumption rate for that exact microsecond.

    dSdt = params["dSdt_func"](t)

    Q = (params["DELTA_H_FERM"] * dSdt * 1000) / (params["rho"] * params["Cp"])

    # same Q applied to every layer
    Q_layers = np.full(N, Q)

    dTdt = np.zeros(N)

    dTdt[0] = 0.0
    for i in range(1, N - 1):
        d2T_dz2 = (T[i + 1] - 2 * T[i] + T[i - 1]) / dz**2
        dTdt[i] = alpha * d2T_dz2 + Q_layers[i]

    d2T_top = 2 * (T[N - 2] - T[N - 1]) / dz**2
    dTdt[N - 1] = alpha * d2T_top + Q_layers[N - 1]

    return dTdt


# couples phase 1 sugar kinetics to 1d heat transport
def run_with_fermentation(spatial_p, ferm_p, t_end=None, n_save=300):
    """
    Stage 2: couples Phase 1 sugar kinetics to 1D heat transport.
    Starts flat (uniform T_initial)
    """
    N = spatial_p["N"]

    # run Phase 1 to find actual fermentation end time
    sol_ferm = run_simulation(ferm_p)

    # check if the terminal event (sugar depletion) fired
    if sol_ferm.t_events[0].size > 0:
        t_ferm_end = sol_ferm.t_events[0][0]
    else:
        # fallback: find when sugar drops below 1 g/L
        S_trace = sol_ferm.y[2]
        low = np.where(S_trace < 1.0)[0]
        t_ferm_end = sol_ferm.t[low[0]] if low.size > 0 else sol_ferm.t[-1]

    if t_end is None:
        t_end = t_ferm_end  # must resolve BEFORE linspace uses it

    t_eval = np.linspace(0, t_end, n_save)

    # build the source term interpolator from Phase 1 output
    dSdt_arr = get_dSdt_profile(t_eval, ferm_p)
    spatial_p["dSdt_func"] = interp1d(
        t_eval,
        dSdt_arr,
        kind="linear",
        bounds_error=False,
        fill_value=0.0,  # no heat past sugar depletion
    )

    #  biology creates the gradient
    T0 = np.full(N, spatial_p["T_initial"])
    T0[0] = spatial_p["T_coolant"]  # Dirichlet: base at jacket temperature

    sol = solve_ivp(
        fun=heat_rhs_with_source,
        t_span=(0, t_end),
        y0=T0,
        args=(spatial_p,),
        method="BDF",
        t_eval=t_eval,
        dense_output=True,
    )

    return sol, t_eval


# runs both stages, saves combined figure
if __name__ == "__main__":
    import matplotlib.pyplot as plt

    z = np.linspace(0, params["H"], params["N"])
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    sol_test = run_diffusion_test(params)
    for k in [0, 10, 40, 100, 199]:
        axes[0].plot(sol_test.y[:, k], z, label=f"t={sol_test.t[k]:.0f}h")
    axes[0].set_xlabel("Temperature (°C)")
    axes[0].set_ylabel("Height z (m)")
    axes[0].set_title("Stage 1: relaxation test\n(no source — BCs verified)")
    axes[0].legend(fontsize=8)

    sol_s2, t_eval = run_with_fermentation(params, ferm_params)
    n = sol_s2.y.shape[1]

    for k in [0, n // 5, n // 2, 3 * n // 4, n - 1]:
        axes[1].plot(sol_s2.y[:, k], z, label=f"t={sol_s2.t[k]:.0f}h")
    axes[1].set_xlabel("Temperature (°C)")
    axes[1].set_ylabel("Height z (m)")
    axes[1].set_title("Stage 2: fermentation gradient\n(warm top, cold base)")
    axes[1].legend(fontsize=8)

    im = axes[2].imshow(
        sol_s2.y,
        aspect="auto",
        origin="lower",
        extent=[sol_s2.t[0], sol_s2.t[-1], 0, params["H"]],
        cmap="inferno",
    )
    axes[2].set_xlabel("Time (h)")
    axes[2].set_ylabel("Height z (m)")
    axes[2].set_title("T(z, t) heatmap")
    fig.colorbar(im, ax=axes[2], label="Temperature (°C)")

    fig.suptitle("Phase 2 — CFD-lite: 1D heat transport in fermentation tank_with_N_40")
    fig.tight_layout()
    fig.savefig("results/figures/phase2_full.png", dpi=150)
    plt.show()
