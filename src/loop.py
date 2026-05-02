import json 

from itertools import product
import pandas as pd 

from toolbox import (
    LoopConfig, 
    CustomLogger,
    extract_hyperparameters, 
    sanitize_df, 
    to_saving_logs, 
    already_done
)
from single_run import single_run

TEST_MODE = True
DEVICE_BATCH_SIZE = 4
logger = CustomLogger("./custom_logs")

def loop():
    with open("./config_files/config-loop.json") as file:
        config_json = json.load(file)

    parameter_names, parameters_values = extract_hyperparameters(config_json)
    print(parameter_names)
    for dataset_info in config_json["datasets"]:
        df = pd.read_csv(dataset_info["filepath-train"])
        df = sanitize_df(df, **dataset_info)
        labels = list(df["LABEL"].unique())
        
        df_prediction = pd.read_csv(dataset_info["filepath-predict"])
        df_prediction = sanitize_df(df_prediction, **dataset_info)
        for label in labels: 
            task_name = f"{dataset_info['name']}-{label}"
            for local_config in product(*parameters_values):
                loop_config = LoopConfig(
                    task_name=task_name,
                    dichotomization_label = label,
                    **{n: v for n,v in zip(parameter_names,local_config)},
                    test_mode = TEST_MODE, 
                    device_batch_size = DEVICE_BATCH_SIZE,
                )
                logger.start_loop_log(loop_config)
                if already_done(loop_config):
                    logger("Loop already done, skipping")
                else:   
                    hash_, to_save = single_run(df, df_prediction, loop_config)
                    print(f"HASH: {hash_}")
                    print(to_save)
                    to_saving_logs(hash_, to_save)
                logger("END LOOP" + "#" * 92)

                break
            break
        break 

if __name__ == "__main__":
    loop()
