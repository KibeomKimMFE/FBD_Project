import os


def get_file_date(path: str) -> str:
    file_name = os.path.split(path)[1]
    next_date = file_name.split("_")[-2]
    return next_date
