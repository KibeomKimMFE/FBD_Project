import numpy as np
import pandas as pd

from numpy.linalg import inv
from scipy.linalg import block_diag


def extract_features(orderbook_file_path: str) -> pd.DataFrame:
    """
    reads csv file of given path and extracts level 1 orderbook data.
    Then it calculates mid price, bid-ask spread and imbalance.

    Args:
        orderbook_file_path (str): path of the csv file.

    Returns:
        pd.DataFrame: level 1 orderbook data.
    """
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


def preprocess_data(
    df: pd.DataFrame, numSpreads: int = 4, numImbalance: int = 4
) -> pd.DataFrame:
    """
    this function converts the spread and imbalance value into discretized
    value and calculates difference of mid prices using current and next value.
    Also, it adds symmetrized data which is discussed in Stoikov(2018).

    Args:
        df (pd.DataFrame): dataset containing bid,ask and its volumes.
        numSpreads (int, optional): number of spreads to discretize.
        numImbalance (int, optional): number of imbalance to discretize.

    Returns:
        pd.DataFrame: symmetrized dataset.
    """

    df_signal = df.copy(deep=True)
    tick_size = df_signal.ba_spread[df_signal.ba_spread != 0].min()

    # discretize bidask spread then get next time's bidask spread
    # discretize imbalance and get next imbalance
    # cap spread values that goes over 0.2 as 0.25 (group 5 means spread is over 0.2)
    df_signal = df_signal[df_signal.ba_spread <= numSpreads * tick_size]
    df_signal["ba_spread"] = np.round(df_signal["ba_spread"].div(tick_size)).astype(int)
    df_signal["imbalance"] = pd.cut(
        df["imbalance"],
        bins=np.arange(numImbalance + 1) / numImbalance,
        labels=np.arange(1, numImbalance + 1),
    ).astype(int)

    # calculate change in mid price
    # include data that bidask spread is within 0.2, same goes for
    # mid price change
    df_signal["mid_chg"] = np.round(df_signal["mid_price"].diff(), 2).shift(
        -1,
    )
    df_signal = df_signal[abs(df_signal.mid_chg) <= 0.1]

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


def get_micro_adjustment(df_sym: pd.DataFrame) -> list[np.ndarray, np.ndarray]:
    """
    calculates micro price adjustment g1 and B given
    symmetrized dataset containing current and next step's (imbalance and spread).
    Note that imbalance and spread should be 'discretized' before running.

    Args:
        df_sym (pd.DataFrame): symmetrized dataset containnig

    Returns:
        g1, B (list[np.ndarray, np.ndarray]): micro price adjustments. See the Stoikov (2018).
    """

    # find the number of imbalance, spread and these combinations.
    nSpread, nImbalance = len(df_sym.ba_spread.unique()), len(df_sym.imbalance.unique())
    nCombination = nSpread * nImbalance

    # divide datafrmae into two events (dM equals to 0 or not equal to 0)
    mid_zero, mid_non_zero = (
        df_sym[df_sym["mid_chg"] == 0],
        df_sym[df_sym["mid_chg"] != 0],
    )

    # transition matrix Q
    mid_zero = mid_zero.groupby(["ba_spread", "imbalance", "next_imbalance"])[
        "mid_price"
    ].count()
    Q_cnt = pd.DataFrame(
        [],
        index=pd.MultiIndex.from_product(
            [
                list(range(1, nSpread + 1)),
                list(range(1, nImbalance + 1)),
                list(range(1, nImbalance + 1)),
            ],
            names=["ba_spread", "imbalance", "next_imbalance"],
        ),
        columns=["cnt"],
    ).fillna(0)

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
        .unstack("mid_chg")
    )
    R = R["mid_price"].fillna(0).values

    # get transition matrix (transient, absorbing state)
    J = np.concatenate([Q_cnt, R], axis=1)

    # calculate probability
    J = J / J.sum(axis=1).reshape(-1, 1)
    J = np.nan_to_num(J, nan=0)

    # split Q, R and define K
    Q, R = J[:, :nCombination], J[:, nCombination:]
    I, K = np.eye(nCombination), np.array([-0.1, -0.05, 0.05, 0.1]).reshape(-1, 1)

    # 1st order micro-price adjustment
    g1 = inv(I - Q) @ R @ K

    # define new absorbing state
    T_cnt = mid_non_zero.pivot_table(
        index=["ba_spread", "imbalance"],
        columns=["next_ba_spread", "next_imbalance"],
        values="mid_price",
        fill_value=0,
        aggfunc="count",
    ).values

    J2 = np.concatenate([Q_cnt, T_cnt], axis=1)
    J2 = J2 / J2.sum(axis=1).reshape(-1, 1)

    # new Q and T
    Q, T = J2[:, :nCombination], J2[:, nCombination:]

    # calculate B matrix
    B = inv(I - Q) @ T
    return g1, B
