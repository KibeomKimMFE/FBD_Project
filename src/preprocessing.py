import numpy as np
import pandas as pd

from numpy.linalg import inv
from scipy.linalg import block_diag


def extract_features(orderbook_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(orderbook_file_path)[
        [
            "timestamp",
            "asks[0].price",
            "bids[0].price",
            "asks[0].amount",
            "bids[0].amount",
        ]
    ]

    # calculate mid price and bidask spread
    df["mid_price"] = (df["asks[0].price"] + df["bids[0].price"]) / 2
    df["ba_spread"] = np.round((df["asks[0].price"] - df["bids[0].price"]), 2)
    df["imbalance"] = df["bids[0].amount"] / (
        df["bids[0].amount"] + df["asks[0].amount"]
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms")

    # convert timestamp to datetime format
    df = df[["timestamp", "mid_price", "ba_spread", "imbalance"]].set_index("timestamp")

    # resample by 1second frequency
    df = df.resample("1s").last().ffill()
    return df


def symmetrize_data(
    df_feature: pd.DataFrame, numSpreads: int = 4, numImbalance: int = 4, numdM: int = 2
) -> pd.DataFrame:
    """_summary_

    Args:
        df_feature (pd.DataFrame): _description_
        numSpreads (int, optional): _description_. Defaults to 4.
        numImbalance (int, optional): _description_. Defaults to 4.

    Returns:
        pd.DataFrame: _description_
    """

    df_signal = df_feature.copy(deep=True)
    tick_size = df_signal.ba_spread[df_signal.ba_spread != 0].min()

    # discretize bidask spread then get next time's bidask spread
    # discretize imbalance and get next imbalance
    df_signal = df_signal[df_signal.ba_spread <= numSpreads * tick_size]
    df_signal["ba_spread"] = np.round(df_signal["ba_spread"].div(tick_size)).astype(int)
    df_signal["imbalance"] = pd.cut(
        df_feature["imbalance"],
        bins=np.arange(numImbalance + 1) / numImbalance,
        labels=np.arange(1, numImbalance + 1),
    ).astype(int)

    # calculate change in mid price
    # include data that bidask spread is within 0.2, same goes for
    # mid price change
    df_signal["mid_chg"] = (
        np.round(df_signal["mid_price"].diff().div(tick_size))
        .mul(tick_size)
        .shift(
            -1,
        )
    )
    df_signal = df_signal[abs(df_signal.mid_chg) <= tick_size * numdM]

    df_signal["next_ba_spread"] = df_signal["ba_spread"].shift(-1)
    df_signal["next_imbalance"] = df_signal["imbalance"].shift(-1)
    df_signal = df_signal.dropna()

    # make symmetric data
    df_symmetric = df_signal.copy(deep=True)
    df_symmetric["imbalance"] = numImbalance - df_signal["imbalance"] + 1
    df_symmetric["next_imbalance"] = numImbalance - df_signal["next_imbalance"] + 1
    df_symmetric["mid_chg"] = -df_signal["mid_chg"]

    df = pd.concat([df_signal, df_symmetric])
    df[["next_ba_spread", "next_imbalance"]] = df[
        ["next_ba_spread", "next_imbalance"]
    ].astype(int)
    return df.dropna()


def get_micro_adjustment(df_sig: pd.DataFrame) -> list[np.ndarray, np.ndarray]:
    """_summary_

    Args:
        df_sig (pd.DataFrame): _description_

    Returns:
        list[np.ndarray, np.ndarray]: _description_
    """
    nSpread, nImbalance = len(df_sig.ba_spread.unique()), len(df_sig.imbalance.unique())
    nCombination = nSpread * nImbalance

    # divide datafrmae into two events (dM equals to 0 or not equal to 0)
    mid_zero, mid_non_zero = (
        df_sig[df_sig["mid_chg"] == 0],
        df_sig[df_sig["mid_chg"] != 0],
    )

    print(mid_zero.shape, mid_non_zero.shape)
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
    return g1, B
