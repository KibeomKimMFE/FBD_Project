import dask
import numpy as np
import pandas as pd


@dask.delayed
def extract_features(orderbook_file_path: str) -> pd.DataFrame:
    """
    Given the orderbook file path, this function extracts
    key information for computing micro price predictions.
    First, it calculates bid-ask spreaad, mid price, bid-ask imbalance.
    Note that this function resamples above quantities by 1 second frequency.

    Args:
        orderbook_file_path (str): orderbook_file path

    Returns:
        pd.DataFrame: a dataframe containing necessary quantities.
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
    df["ba_spread"] = np.round((df["asks[0].price"] - df["bids[0].price"]), 5)
    df["imbalance"] = df["bids[0].amount"] / (
        df["bids[0].amount"] + df["asks[0].amount"]
    )

    df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms")

    # convert timestamp to datetime format
    df = df[["timestamp", "mid_price", "ba_spread", "imbalance"]].set_index("timestamp")

    # resample by 1second frequency
    df = df.resample("1s").last().ffill()
    return df


@dask.delayed
def extract_quotes(trade_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(trade_file_path)[
        ["timestamp", "ask_price", "bid_price", "ask_amount", "bid_amount"]
    ]
    df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms")
    return df.set_index("timestamp")


@dask.delayed
def extract_trades(trade_file_path: str) -> pd.DataFrame:
    df = pd.read_csv(trade_file_path)[["timestamp", "side", "price", "amount"]]
    df["timestamp"] = pd.to_datetime(df["timestamp"] / 1000, unit="ms")
    return df.set_index("timestamp")


def symmetrize_data(
    df_feature: pd.DataFrame,
    numSpreads: int = 4,
    numImbalance: int = 4,
    numdM: int = 2,
    symmetrize: bool = True,
) -> pd.DataFrame:
    """_summary_

    Args:
        df_feature (pd.DataFrame): _description_
        numSpreads (int, optional): _description_. Defaults to 4.
        numImbalance (int, optional): _description_. Defaults to 4.
        numdM (int, optional): _description_. Defaults to 2.
        symmetrize (bool, optional): _description_. Defaults to True.

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

    if symmetrize:  # make symmetric data
        df_symmetric = df_signal.copy(deep=True)
        df_symmetric["imbalance"] = numImbalance - df_signal["imbalance"] + 1
        df_symmetric["next_imbalance"] = numImbalance - df_signal["next_imbalance"] + 1
        df_symmetric["mid_chg"] = -df_signal["mid_chg"]
        df = pd.concat([df_signal, df_symmetric])
    else:
        df = df_signal

    df[["next_ba_spread", "next_imbalance"]] = df[
        ["next_ba_spread", "next_imbalance"]
    ].astype(int)
    return df.dropna()
