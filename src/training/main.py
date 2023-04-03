import os
from pathlib import Path

import torch
import whisper
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torch import nn
from tqdm import tqdm

# from NST_dk_train import NST_dk
from common_voice_train import Common_voice_11
from train import Config, WhisperModelModule

BATCH_SIZE = 2
TRAIN_RATE = 0.8

AUDIO_MAX_LENGTH = 480000
TEXT_MAX_LENGTH = 120
SEED = 3407
DEVICE = "gpu" if torch.cuda.is_available() else "cpu"

pwd = os.getcwd()


def main():
    seed_everything(SEED, workers=True)

    log_output_dir = pwd + "/content/logs"
    check_output_dir = pwd + "/content/artifacts"

    train_name = "whisper"
    train_id = "00001"

    model_name = "base"
    lang = "da"

    woptions = whisper.DecodingOptions(language="da", without_timestamps=True)
    wtokenizer = whisper.tokenizer.get_tokenizer(
        True, language="da", task=woptions.task
    )

    cfg = Config()

    tflogger = TensorBoardLogger(
        save_dir=log_output_dir, name=train_name, version=train_id
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=f"{check_output_dir}/checkpoint",
        filename="checkpoint-{epoch:04d}",
        save_top_k=-1,  # all model save
    )

    callback_list = [checkpoint_callback, LearningRateMonitor(logging_interval="epoch")]

    trainer = Trainer(
        precision=16,
        accelerator=DEVICE,
        max_epochs=cfg.num_train_epochs,
        accumulate_grad_batches=cfg.gradient_accumulation_steps,
        logger=tflogger,
        callbacks=callback_list
    )

    train_set = Common_voice_11(wtokenizer, split="train")
    eval_set = Common_voice_11(wtokenizer, split="test")
    model = WhisperModelModule(cfg, wtokenizer, train_set, eval_set, model_name, lang)

    trainer.fit(model)


if __name__ == "__main__":
    main()
