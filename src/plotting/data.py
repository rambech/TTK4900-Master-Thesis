import os
from datetime import datetime
import json


def save_data(data: dict, name: str) -> None:
    log_dir = "log_data/logs"
    assert (os.path.exists(log_dir)), f"{log_dir} does not exist"
    current_time = datetime.now()
    log_name = name + current_time.strftime("%m-%d-h-M")
    file_name = f"{log_name}.json"
    log_path = os.path.join(log_dir, file_name)

    with open(log_path, "w") as json_file:
        json.dump(data, json_file, indent=2)
