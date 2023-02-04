"""
this file downloads raw orderbook, price, trade data 
using tardis.dev API. This is added just for the sake of
showing how we downloaded the data.

for more about this API, please go https://docs.tardis.dev
"""
from tardis_dev import datasets


if __name__ == "__main__":
    # we download crypto futures data for the period
    # 2nd September 2022 to 31st December 2022. (only allowed for academic purposes)
    datasets.download(
        exchange="binance-futures",
        data_types=["trades", "quotes", "book_snapshot_5"],
        from_date="2022-09-02",
        to_date="2022-12-31",
        symbols=["adausdt"],
        api_key="",  # this info needs to be included to use this package.
    )
