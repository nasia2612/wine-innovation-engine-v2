from src.process.simulator import rhs, params, STATE_NAMES


def test_rhs_runs():
    # smoke test ,to check if rhs runs without erros and shows 5 derivatives
    # it makes a state vector y0 with the initial conditions from params and calls rhs with t=0, y=y0, p=params
    y0 = [params["X0"], params["Xt0"], params["S0"], params["N0"], params["E0"]]
    # you cal the rhs to get the derivatives
    derivs = rhs(0, y0, params)
    # you assert that the length of the derivatives is 5,IF NOT IT WILL RAISE AN ERROR
    assert len(derivs) == len(STATE_NAMES)


def test_rhs_signs():
    # sanity test  to check if the signs of the derivatives make sense at the start of fermentation
    y0 = [params["X0"], params["Xt0"], params["S0"], params["N0"], params["E0"]]
    dX, dXt, dS, dN, dE = rhs(0, y0, params)
    # at the start of fermentation, we expect biomass to grow (dX>0), sugar and nitrogen to decrease (dS<0, dN<0)
    # and ethanol to increase (dE>0)
    assert dX > 0, "Biomass should grow at the start of fermentation"
    assert dS < 0, "Sugar should decrease at the start of fermentation"
    assert dN < 0, "Nitrogen should decrease at the start of fermentation"
    assert dE > 0, "Ethanol should increase at the start of fermentation"
