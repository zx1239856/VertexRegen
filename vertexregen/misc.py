import json
import os
from transformers import TrainerCallback, TrainerState
from transformers.trainer_utils import get_last_checkpoint as hf_get_last_checkpoint


class JsonlLoggerCallback(TrainerCallback):
    def __init__(self, log_file_path):
        self.log_file_path = os.path.join(log_file_path, "log.jsonl")

    def on_log(self, args, state: TrainerState, control, logs=None, **kwargs):
        if logs is None:
            return
        _ = logs.pop("total_flos", None)
        if state.is_world_process_zero:
            log_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                **logs,
            }

            with open(self.log_file_path, "a") as f:
                f.write(json.dumps(log_entry) + "\n")


def get_last_checkpoint(output_dir: str) -> str | None:
    if not os.path.exists(output_dir):
        return
    last_ckpt = hf_get_last_checkpoint(output_dir)
    return last_ckpt
