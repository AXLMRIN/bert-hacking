import sys
import pandas as pd

from config import load_config
from environment import prepare_env
from evaluate import compute_metrics_multiclass
from experiment import create_hash
from model import train, predict
from preprocess import sanitize_df, dichotomize
from splits import split_ds


def main():
    config_path = sys.argv[1]
    config = load_config(config_path)

    data_config = config["data"]
    split_config = config["split"]
    model_config = config["model"]
    training_config = config["training"]

    prepare_env([
        training_config["output_dir"],
    ])

    df = pd.read_csv(data_config["input_path"])

    df = sanitize_df(
        df,
        text_column=data_config["text_column"],
        label_column=data_config["label_column"],
        id_column=data_config.get("id_column"),
        drop_na_text=data_config.get("drop_na_text", True),
        drop_na_label=data_config.get("drop_na_label", True),
        enforce_unique_id=data_config.get("enforce_unique_id", False),
    )

    if data_config.get("dichotomize", False):
        df = dichotomize(
            df,
            threshold=data_config["dichotomize_threshold"],
            positive_if=data_config.get("positive_if", "greater"),
            label_column="LABEL",
        )

    splits = split_ds(
        df,
        train_size=split_config.get("train_size", 0.8),
        validation_size=split_config.get("validation_size", 0.1),
        test_size=split_config.get("test_size", 0.1),
        random_state=split_config.get("random_state", 42),
        stratify=split_config.get("stratify", False),
        label_column="LABEL",
    )

    train_df = splits["train"]
    validation_df = splits["validation"]
    test_df = splits["test"]

    run_hash = create_hash({
        "model_name": model_config["model_name"],
        "train_size": split_config.get("train_size", 0.8),
        "validation_size": split_config.get("validation_size", 0.1),
        "test_size": split_config.get("test_size", 0.1),
        "learning_rate": training_config.get("learning_rate", 2e-5),
        "batch_size": training_config.get("per_device_train_batch_size", 8),
        "epochs": training_config.get("num_train_epochs", 3),
        "seed": training_config.get("seed", 42),
    })

    print("Run hash:", run_hash)
    print("Train size:", len(train_df))
    print("Validation size:", len(validation_df) if validation_df is not None else 0)
    print("Test size:", len(test_df) if test_df is not None else 0)

    training_output = train(
        train_df=train_df,
        validation_df=validation_df,
        model_name=model_config["model_name"],
        output_dir=training_config["output_dir"],
        compute_metrics=compute_metrics_multiclass,
        text_column="TEXT",
        label_column="LABEL",
        num_labels=model_config.get("num_labels"),
        max_length=model_config.get("max_length"),
        learning_rate=training_config.get("learning_rate", 2e-5),
        per_device_train_batch_size=training_config.get("per_device_train_batch_size", 8),
        per_device_eval_batch_size=training_config.get("per_device_eval_batch_size", 8),
        num_train_epochs=training_config.get("num_train_epochs", 3),
        weight_decay=training_config.get("weight_decay", 0.0),
        evaluation_strategy=training_config.get("evaluation_strategy", "epoch"),
        save_strategy=training_config.get("save_strategy", "epoch"),
        logging_strategy=training_config.get("logging_strategy", "epoch"),
        load_best_model_at_end=training_config.get("load_best_model_at_end", True),
        metric_for_best_model=training_config.get("metric_for_best_model", "f1"),
        greater_is_better=training_config.get("greater_is_better", True),
        save_total_limit=training_config.get("save_total_limit", 1),
        seed=training_config.get("seed", 42),
        trust_remote_code=model_config.get("trust_remote_code", False),
    )

    if test_df is not None and len(test_df) > 0:
        predictions = predict(
            df=test_df,
            model=training_output["model"],
            tokenizer=training_output["tokenizer"],
            text_column="TEXT",
            id_column="ID" if "ID" in test_df.columns else None,
            label_column="LABEL",
            max_length=model_config.get("max_length"),
            batch_size=training_config.get("per_device_eval_batch_size", 8),
        )

        predictions_path = (
            f"{training_config['output_dir']}/test_predictions_{run_hash}.csv"
        )
        predictions.to_csv(predictions_path, index=False)
        print("Saved test predictions to:", predictions_path)


if __name__ == "__main__":
    main()
