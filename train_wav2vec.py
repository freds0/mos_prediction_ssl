import numpy as np
from transformers import EvalPrediction
from transformers import AutoConfig, Wav2Vec2Processor

from src.models import Wav2Vec2ForSpeechClassification
from src.collator import DataCollatorCTCWithPadding
from src.trainer import CTCTrainer

# Loading the created dataset using datasets
from datasets import load_dataset, load_metric

import torchaudio
import torch

from transformers import TrainingArguments

#import ast

def str2list(batch):
    batch["score"] = [batch["score"]]
    return batch

data_files = {
    "train": "./dataset/train.csv",
    "validation": "./dataset/test.csv",
}

dataset = load_dataset("csv", data_files=data_files, delimiter="\t")
dataset = dataset.map(str2list)
train_dataset = dataset["train"]
eval_dataset = dataset["validation"]

print('Dataset Info:')
print(train_dataset)
print(eval_dataset)

# We need to specify the input and output column
input_column = "path"
output_column = "score"

num_labels = len(train_dataset[0][output_column])
label_list = list(range(num_labels))
print(f"A regression problem with {num_labels} items: {label_list}")
is_regression = True

# Verify: facebook/wav2vec2-xls-r-1b
model_name_or_path = "facebook/wav2vec2-large-960h-lv60-self"
pooling_mode = "mean"

# config
config = AutoConfig.from_pretrained(
    model_name_or_path,
    num_labels=num_labels,
    label2id={label: i for i, label in enumerate(label_list)},
    id2label={i: label for i, label in enumerate(label_list)},
    finetuning_task="wav2vec2_clf",
    problem_type="regression"
)
setattr(config, 'pooling_mode', pooling_mode)

processor = Wav2Vec2Processor.from_pretrained(model_name_or_path,)
target_sampling_rate = processor.feature_extractor.sampling_rate
print(f"The target sampling rate: {target_sampling_rate}")

def speech_file_to_array_fn(path):
    speech_array, sampling_rate = torchaudio.load(path)
    resampler = torchaudio.transforms.Resample(sampling_rate, target_sampling_rate)
    speech = resampler(speech_array).squeeze().numpy()
    return speech

def preprocess_function(examples):
    speech_list = [speech_file_to_array_fn(path) for path in examples[input_column]]
    target_list = [label for label in examples[output_column]] # Do any preprocessing on your float/integer data

    result = processor(speech_list, sampling_rate=target_sampling_rate)
    result["labels"] = list(target_list)

    return result

print("Preprocessing Train Dataset...")

train_dataset = train_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)

print("Preprocessing Test Dataset...")
eval_dataset = eval_dataset.map(
    preprocess_function,
    batch_size=100,
    batched=True,
    num_proc=4
)
print("Creating Model...")

data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)

def compute_metrics(p: EvalPrediction):
    preds = p.predictions[0] if isinstance(p.predictions, tuple) else p.predictions
    preds = np.squeeze(preds) if is_regression else np.argmax(preds, axis=1)

    if is_regression:
        return {"mse": ((preds - p.label_ids) ** 2).mean().item()}
    else:
        return {"accuracy": (preds == p.label_ids).astype(np.float32).mean().item()}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

model = Wav2Vec2ForSpeechClassification.from_pretrained(
    model_name_or_path,
    config=config,
)

model.freeze_feature_extractor()

training_args = TrainingArguments(
    output_dir="./wav2vec2-xlsr-mos-prediction",
    # output_dir="/content/gdrive/MyDrive/wav2vec2-xlsr-greek-speech-emotion-recognition"
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=1,
    evaluation_strategy="steps",
    num_train_epochs=1.0,
    fp16=True,
    save_steps=1000,
    eval_steps=1000,
    logging_steps=1000,
    learning_rate=1e-4,
    save_total_limit=2,
)

trainer = CTCTrainer(
    model=model,
    data_collator=data_collator,
    args=training_args,
    compute_metrics=compute_metrics,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
)

trainer.train()