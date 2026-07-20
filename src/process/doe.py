# phase 3- box behnken design generation + phase 1 simulator runs
# we extract a csv for the DoE/RSM/Desirability pipeline for R to do the fit

import numpy as np
import pandas as pd
from pyDOE3 import bbdesign
from pathlib import Path

from src.process.simulator import run_simulation, params

# we write the factor levels with symmetric levels because we convert them into coded variables (we want to put every factor on a common,comparable footing)
FACTOR_LEVELS = {
    "T": {"low": 15, "center": 21, "high": 27},  # degC
    "S0": {"low": 250, "center": 265, "high": 280},  # g/L
    "N0": {"low": 140, "center": 240, "high": 340},  # mg/L
}


def decode(
    coded_value: float, factor: str
) -> float:  # it is expected to return a float
    low = FACTOR_LEVELS[factor]["low"]
    center = FACTOR_LEVELS[factor]["center"]
    step = center - low
    return (
        center + coded_value * step
    )  # the coding equation is coded_value=(actual_value-center)/step, so we invert it to get actual_value=center+coded_value*step


# BUILD THE BOX-BEHNKEN DESIGN


def build_design() -> pd.DataFrame:
    coded = bbdesign(
        len(FACTOR_LEVELS), center=3
    )  # center=1 means we have 1 center point, which is the default for bbdesign,we dont need 3 center because  our simulator is deterministic, so this becomes redundant but we keep it for methodological consistency
    df = pd.DataFrame(coded, columns=["x1", "x2", "x3"])
    df["T"] = df["x1"].apply(lambda v: decode(v, "T"))
    df["S0"] = df["x2"].apply(lambda v: decode(v, "S0"))
    df["N0"] = df["x3"].apply(lambda v: decode(v, "N0"))
    return df


# so we run one point to see
DRYNESS_THRESHOLD = 4.0  # g/L
T_END_PENALTY = 1000  # hours, if we reach this time without reaching dryness, we will penalize the objective function


def run_single_point(T: float, S0: float, N0: float) -> dict:
    # override parameters with the DoE point
    p = {**params, "T": T, "S0": S0, "N0": N0}
    sol = run_simulation(
        p=p, t_end=T_END_PENALTY, use_event=False
    )  # we use the penalty time as t_end, so that we can see if we reach dryness or not

    X, Xt, S, N, E = sol.y

    # --- basic responses (unchanged) ---
    final_ethanol = E[-1]
    residual_sugar = max(S[-1], 0.0)
    conversion = (S0 - residual_sugar) / S0

    # --- old absolute-threshold criterion, kept as diagnostic ---
    dry_idx = np.argmax(S < DRYNESS_THRESHOLD) if np.any(S < DRYNESS_THRESHOLD) else -1

    # --- NEW: rate-based practical end-of-fermentation ---
    # dense output lets us evaluate S at any t within the solved window
    t_fine = np.linspace(0, sol.t[-1], 2000)
    S_fine = sol.sol(t_fine)[2]  # index 2 = S in [X, Xt, S, N, E]

    dSdt = np.gradient(S_fine, t_fine)
    rate = np.abs(dSdt)

    peak_idx = np.argmax(rate)
    peak_rate = rate[peak_idx]
    threshold = 0.01 * peak_rate

    # search only AFTER the peak — ignore early ramp-up
    rate_after_peak = rate[peak_idx:]
    below = np.where(rate_after_peak < threshold)[0]

    if below.size > 0:
        j = below[0]  # index inside the sliced array
        real_idx = peak_idx + j  # index inside the ORIGINAL t_fine
        time_to_dry = t_fine[real_idx]
        reached_practical_end = True
    else:
        time_to_dry = T_END_PENALTY
        reached_practical_end = False

    return {
        "ethanol": final_ethanol,
        "sugar": residual_sugar,
        "time": time_to_dry,
        "conversion": conversion,
        "peak_rate": peak_rate,  # g/L/h — peak cooling & CO2 demand
        "time_to_peak": t_fine[peak_idx],  # h — lag phase indicator
        "reached_dryness": dry_idx != -1,  # old absolute-threshold flag
        "reached_practical_end": reached_practical_end,  # new rate-based flag
    }


# we run all the points in the design and return a dataframe with the results


def run_design(design_df: pd.DataFrame) -> pd.DataFrame:
    responses = []
    for (
        _,
        row,
    ) in (
        design_df.iterrows()
    ):  # it loops through the rows ,we use _ because we don't need the index, we just need the row
        # it makes a dict-like object with the row values, and we can access the values by column name
        result = run_single_point(row["T"], row["S0"], row["N0"])
        responses.append(
            result
        )  # for every row we append the result to the responses list, which will be a list of dicts
    responses_df = pd.DataFrame(responses)  # from dict to dataframe
    return pd.concat(
        [design_df, responses_df], axis=1
    )  # final_result = [design factors] + [responses]


if __name__ == "__main__":
    design = build_design()
    print(f"Box-Behnken design: {len(design)} points")

    full_results = run_design(design)

    output_dir = Path("results/phase3_doe")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "design_matrix.csv"

    full_results.to_csv(output_path, index=False)
    print(f"Saved to {output_path}")
    print(
        full_results[
            [
                "T",
                "S0",
                "N0",
                "ethanol",
                "sugar",
                "conversion",
                "peak_rate",
                "time",
                "reached_dryness",
            ]
        ]
    )

    n_stuck = (~full_results["reached_dryness"]).sum()
    if n_stuck > 0:
        print(
            f"\nℹ️  {n_stuck}/{len(full_results)} points did not reach the "
            f"{DRYNESS_THRESHOLD} g/L dryness threshold (nitrogen-limited); "
            f"practical end-of-fermentation time reported instead."
        )
