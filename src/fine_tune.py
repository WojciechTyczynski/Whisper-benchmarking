import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import datasets
import evaluate
import hydra
import torch
from datasets import Audio, DatasetDict, DownloadConfig, load_dataset
from huggingface_hub import login
from loguru import logger
from transformers import (Seq2SeqTrainer, Seq2SeqTrainingArguments,
                          WhisperFeatureExtractor,
                          WhisperForConditionalGeneration, WhisperProcessor,
                          WhisperTokenizer)

# datasets.config.DOWNLOADED_DATASETS_PATH = Path('/work3/s212373/datasets')





@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def load_common_voice():
    # download_config = DownloadConfig(cache_dir='/work3/s212373/datasets')

    common_voice = DatasetDict()
    common_voice["train"] = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "da", split="train+validation", use_auth_token=True, cache_dir='/work3/s212373/datasets2')
    common_voice["test"] = datasets.load_dataset("mozilla-foundation/common_voice_11_0", "da", split="test", use_auth_token=True, cache_dir='/work3/s212373/datasets2')
    common_voice = common_voice.remove_columns(["accent", "age", "client_id", "down_votes", "gender", "locale", "path", "segment", "up_votes"])
    common_voice = common_voice.cast_column("audio", Audio(sampling_rate=16000))
    return common_voice

def train(cfg, dataset, model, data_collator, processor, compute_metrics):
    
    training_args = Seq2SeqTrainingArguments(
    output_dir=cfg.output_dir,  # change to a repo name of your choice
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,  # increase by 2x for every 2x decrease in batch size
    learning_rate=1e-5,
    warmup_steps=500,
    max_steps=4000,
    gradient_checkpointing=True,
    fp16=True,
    evaluation_strategy="steps",
    per_device_eval_batch_size=8,
    predict_with_generate=True,
    generation_max_length=225,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=25,
    report_to=["tensorboard"],
    load_best_model_at_end=True,
    metric_for_best_model="wer",
    greater_is_better=False,
    push_to_hub=True,
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )

    processor.save_pretrained(training_args.output_dir)
    trainer.train()

    kwargs = {
    "dataset_tags": "mozilla-foundation/common_voice_11_0",
    "dataset": "Common Voice 11.0",  # a 'pretty' name for the training dataset
    "dataset_args": "config: da, split: test",
    "language": "da",
    "model_name": f"Whisper {cfg.model} - {cfg.language} - Wojty",  # a 'pretty' name for our model
    "finetuned_from": "openai/whisper-{cfg.model} ",
    "tasks": "automatic-speech-recognition",
    "tags": "hf-asr-leaderboard",
    }

    trainer.push_to_hub(**kwargs)


@hydra.main(version_base=None, config_path="../conf", config_name="conf_ft.yaml")
def run(cfg) -> None:
    login(cfg.api_token)
    metric = evaluate.load("wer")

    def compute_metrics(pred):
        
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        
        wer = 100 * metric.compute(predictions=pred_str, references=label_str)
        logger.info(f'Predicted: {pred_str}    True: {label_str}   WER: {wer}')
        return {"wer": wer}
    
    def prepare_dataset(batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array 
        batch["input_features"] = feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids 
        batch["labels"] = tokenizer(batch["sentence"]).input_ids
        return batch

    os.chdir(hydra.utils.get_original_cwd())

    feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/whisper-{cfg.model}")
    tokenizer = WhisperTokenizer.from_pretrained(f"openai/whisper-{cfg.model}", language=cfg.language, task=cfg.task)
    processor = WhisperProcessor.from_pretrained(f"openai/whisper-{cfg.model}", language=cfg.language, task=cfg.task)

    logger.info(f"Loading dataset has started...")
    common_voice = load_common_voice()
    logger.info(f"Preparing dataset...")
    common_voice = common_voice.map(prepare_dataset, remove_columns=common_voice.column_names["train"], num_proc=2)

    logger.info(f"Loading datacollector")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    logger.info(f"Loading {cfg.model} model")
    model = WhisperForConditionalGeneration.from_pretrained(f"openai/whisper-{cfg.model}")
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []

    logger.info(f"Training model...")
    train(cfg, common_voice, model, data_collator, processor, compute_metrics)


if __name__ == "__main__":
    run()