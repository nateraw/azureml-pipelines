import os
from pathlib import Path

from datasets import Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


from transformers.integrations import MLflowCallback
from transformers import logging
import os
logger = logging.get_logger()

class MyCallback(MLflowCallback):
    
    def setup(self, args, state, model):
        """
        Setup the optional MLflow integration.

        Environment:
            HF_MLFLOW_LOG_ARTIFACTS (:obj:`str`, `optional`):
                Whether to use MLflow .log_artifact() facility to log artifacts.

                This only makes sense if logging to a remote server, e.g. s3 or GCS. If set to `True` or `1`, will copy
                whatever is in TrainerArgument's output_dir to the local or remote artifact storage. Using it without a
                remote storage will just copy the files to your artifact location.
        """
        log_artifacts = os.getenv("HF_MLFLOW_LOG_ARTIFACTS", "FALSE").upper()
        if log_artifacts in {"TRUE", "1"}:
            self._log_artifacts = True
        if state.is_world_process_zero:
            self._ml_flow.start_run()
            combined_dict = args.to_dict()
            if hasattr(model, "config") and model.config is not None:
                model_config = model.config.to_dict()
                combined_dict = {**model_config, **combined_dict}
            # remove params that are too long for MLflow
            for name, value in list(combined_dict.items()):
                # internally, all values are converted to str in MLflow
                if len(str(value)) > self._MAX_PARAM_VAL_LENGTH:
                    logger.warning(
                        f"Trainer is attempting to log a value of "
                        f'"{value}" for key "{name}" as a parameter. '
                        f"MLflow's log_param() only accepts values no longer than "
                        f"250 characters so we dropped this attribute."
                    )
                    del combined_dict[name]
            # MLflow cannot log more than 100 values in one go, so we have to split it
            combined_dict_items = list(combined_dict.items())
            for i in range(0, len(combined_dict_items), self._MAX_PARAMS_TAGS_PER_BATCH):
                self._ml_flow.log_params(dict(combined_dict_items[i : i + self._MAX_PARAMS_TAGS_PER_BATCH]))
                break  # HACK - just log 100, no more. AzureML doesn't allow for it. vvv lame.
        self._initialized = True


def main(
    input_dir: str = "./data_dir",
    output_dir: str = './outputs',
    logging_dir: str = './logs',
    num_train_epochs: int = 2,
    model_name_or_path: str = 'distilbert-base-cased',
    seed: int = 42,
):
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_name_or_path, num_labels=2)
    small_train_dataset = Dataset.load_from_disk(Path(input_dir) / 'train')
    small_eval_dataset = Dataset.load_from_disk(Path(input_dir) / 'eval')

    training_args = TrainingArguments(
        output_dir,
        # num_train_epochs=num_train_epochs,  # Using steps instead...
        evaluation_strategy="steps",
        save_strategy="steps",
        save_steps=3,
        eval_steps=3,
        max_steps=3,
        logging_steps=10,
        logging_dir=logging_dir,
        report_to='none',
        seed=seed,
        load_best_model_at_end=True,
        metric_for_best_model='eval_loss',
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=small_train_dataset,
        eval_dataset=small_eval_dataset,
        callbacks=[MyCallback],
        tokenizer=tokenizer
    )

    train_result = trainer.train()

    # Saves model and tokenizer, logs, metrics to output_dir
    trainer.save_model()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()


if __name__ == '__main__':
    import fire

    fire.Fire(main)
