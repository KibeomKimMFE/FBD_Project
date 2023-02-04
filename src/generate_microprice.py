import os
import glob
import numpy as np
import pandas as pd

from src.microprice import get_micro_adjustment
from src.preprocessing import extract_features, symmetrize_data


def get_file_date(path: str) -> str:
    file_name = os.path.split(path)[1]
    next_date = file_name.split("_")[-2]
    return next_date


if __name__ == "__main__":

    # set the path for raw and preprocessed folder
    # set the asset pair that you would like to calculate the microprice
    ASSET = "adausdt"
    WIN_LEN = 5
    RAW_DATA_PATH = "/Users/mac/Desktop/Repos/FBD_Project/datasets/raw/"
    PRC_DATA_PATH = "/Users/mac/Desktop/Repos/FBD_Project/datasets/processed/"

    # check if the asset folder exists in the microprice folder
    # if not create a new one.
    destination_path = PRC_DATA_PATH + f"microprice/{ASSET}"
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    # for each window of specified length, process raw data to get features for calculation.
    print(f"Start microprice calculation. (Asset {ASSET})")

    orderbook_list = sorted(glob.glob(RAW_DATA_PATH + ASSET + "/orderbook/*.csv.gz"))
    for i in range(WIN_LEN, len(orderbook_list)):
        next_date = get_file_date(orderbook_list[i])
        print(
            f"Processing... (date: {next_date}, progress: {i-WIN_LEN}/{len(orderbook_list) - 1 - WIN_LEN})"
        )

        # extract raw data
        all_features = [
            extract_features(path) for path in orderbook_list[i - WIN_LEN : i]
        ]
        df_feat = dask.compute(all_features)[0]
        df_feat = pd.concat(df_feat)

        # symmetrized data (for obtaining microprice)
        df_sym = symmetrize_data(df_feat, symmetrize=True)

        # get micro adjustment and save micro adjustment matrix.
        df_micro = get_micro_adjustment(df_sym)
        df_micro.to_csv(
            PRC_DATA_PATH + f"microprice/{ASSET}/micro_adjustment_{next_date}.csv"
        )

    print("compuation complete.")
