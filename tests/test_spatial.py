# Phase 2 spatial heat model.
import numpy as np
import copy
from src.process.spatial import (
    params,
    run_diffusion_test,
    run_with_fermentation,
    get_dSdt_profile,
)
from src.process.simulator import params as ferm_params

# reminder  X, Xt, S, N, E = sol.y   , in the results the rows are N  layers and the columns time


def test_diffusion_runs():
    sol = run_diffusion_test(params)
    assert sol.success
    assert sol.y.shape[0] == params["N"]


def test_dirichlet_bc_holds():
    # we test that the base node stays pinned at T_coolant the whole time
    sol = run_diffusion_test(params)
    base = sol.y[0, :]  # the N=0 all the time
    assert np.allclose(
        base, params["T_coolant"], atol=1e-6
    )  # will return True if they are close enough


def test_diffusion_relaxes():
    # if we do not have a source the gradient must shrink over time
    sol = run_diffusion_test(params)
    spread_start = sol.y[:, 0].max() - sol.y[:, 0].min()
    spread_end = sol.y[:, -1].max() - sol.y[:, -1].min()
    assert spread_end < spread_start  # this shows us that the gradient has relaxed


def test_fermentation_creates_gradient():
    p = copy.deepcopy(params)
    sol, _ = run_with_fermentation(
        p, ferm_params
    )  # we only need sol and not the timings
    top = sol.y[-1, :]
    base = sol.y[0, :]
    peak_grad = (top - base).max()  # the maximun gradient ever formed
    assert peak_grad > 1.0  # At least 1°C difference


def test_dSdt_is_positive():
    # sugar-substrate consumption magnitude is always positive
    t_eval = np.linspace(0, 300, 100)
    dSdt = get_dSdt_profile(t_eval, ferm_params)
    assert np.all(dSdt >= 0)
