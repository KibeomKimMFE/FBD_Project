from tardis_dev import datasets

datasets.download(
    exchange="binance-futures",
    data_types=["trades", "quotes", "book_snapshot_5"],
    from_date="2022-09-02",
    to_date="2022-12-31",
    symbols=["adausdt"],
    api_key="",
)
