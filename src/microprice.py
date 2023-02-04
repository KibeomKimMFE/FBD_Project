import numpy as np
import pandas as pd

from numpy.linalg import inv
from scipy.linalg import block_diag


def get_micro_adjustment(df_sig: pd.DataFrame, power: int = 6) -> pd.DataFrame:
    """
    Computes microprice adjustments for all (imblance, spread) pairs.
    This function estimates all transition, absorbing state matrices given df_sig and then
    calculates B, g. Given the power input, the final adjustment is calculated by

    microprice = B**power @ g

    Args:
        df_sig (pd.DataFrame): a symmetrized pandas dataframe that contain discretized
                                spread, imbalance and mid price.
        power (int): matrix power of B for microprice adjustment.

    Returns:
        pd.DataFrame: microprice adjustment for all imbalance, spread pairs.
    """
    nSpread, nImbalance = len(df_sig.ba_spread.unique()), len(df_sig.imbalance.unique())
    nCombination = nSpread * nImbalance

    # divide datafrmae into two events (dM equals to 0 or not equal to 0)
    mid_zero, mid_non_zero = (
        df_sig[df_sig["mid_chg"] == 0],
        df_sig[df_sig["mid_chg"] != 0],
    )

    # transition matrix Q
    mid_zero = mid_zero.groupby(["ba_spread", "imbalance", "next_imbalance"])[
        "mid_price"
    ].count()
    Q_cnt = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_product(
            [
                list(range(1, nSpread + 1)),
                list(range(1, nImbalance + 1)),
                list(range(1, nImbalance + 1)),
            ],
            names=["ba_spread", "imbalance", "next_imbalance"],
        ),
        columns=["cnt"],
    )
    Q_cnt.loc[mid_zero.index] = mid_zero.values.reshape(-1, 1)
    Q_cnt = block_diag(
        Q_cnt.loc[1].values.reshape(nSpread, nImbalance),
        Q_cnt.loc[2].values.reshape(nSpread, nImbalance),
        Q_cnt.loc[3].values.reshape(nSpread, nImbalance),
        Q_cnt.loc[4].values.reshape(nSpread, nImbalance),
    )

    # absorbing state matrix R
    R = (
        mid_non_zero.groupby(["ba_spread", "imbalance", "mid_chg"])
        .count()
        .unstack("mid_chg")["mid_price"]
        .fillna(0)
    )
    ordering = R.index

    # K contains all the non-zero mid price chg.
    K = R.columns.values.reshape(-1, 1)

    R_cnt = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_product(
            [list(range(1, nSpread + 1)), list(range(1, nImbalance + 1))],
            names=["ba_spread", "imbalance"],
        ),
        columns=R.columns.values,
    )
    R_cnt.loc[R.index] = R.values
    R_cnt = R_cnt.values

    # get transition prob matrix (transient, absorbing state)
    J = np.concatenate([Q_cnt, R_cnt], axis=1)
    J = np.nan_to_num(J / J.sum(axis=1).reshape(-1, 1), nan=0)

    # split Q, R and define I
    Q, R, I = J[:, :nCombination], J[:, nCombination:], np.eye(nCombination)

    # 1st order micro-price adjustment
    g1 = inv(I - Q) @ R @ K

    # define new absorbing state
    T_cnt = pd.DataFrame(
        0,
        index=pd.MultiIndex.from_product(
            [list(range(1, nSpread + 1)), list(range(1, nImbalance + 1))],
            names=["ba_spread", "imbalance"],
        ),
        columns=pd.MultiIndex.from_product(
            [list(range(1, nSpread + 1)), list(range(1, nImbalance + 1))],
            names=["next_ba_spread", "next_imbalance"],
        ),
    )
    T = mid_non_zero.pivot_table(
        index=["ba_spread", "imbalance"],
        columns=["next_ba_spread", "next_imbalance"],
        values="mid_price",
        fill_value=0,
        aggfunc="count",
    )
    T_cnt.loc[T.index, T.columns] = T.values
    T_cnt = T_cnt.values

    # calculate new trans. prob matrix
    J2 = np.concatenate([Q_cnt, T_cnt], axis=1)
    J2 = np.nan_to_num(J2 / J2.sum(axis=1).reshape(-1, 1), nan=0)

    # new Q and T
    Q, T = J2[:, :nCombination], J2[:, nCombination:]

    # calculate B matrix
    B = inv(I - Q) @ T

    # calculate the micro price adjustment (g_star) based on
    # the matrix power input. (default = 6)
    micro_adj = g1 + np.linalg.matrix_power(B, power) @ g1
    df = pd.DataFrame(micro_adj, index=ordering, columns=["g_star"])
    return df
