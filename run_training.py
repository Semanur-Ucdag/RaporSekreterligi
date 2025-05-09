import json
import torch
import torchaudio
from datasets import Dataset
from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
    TrainingArguments,
    Trainer
)
import transformers
print("Transformers version:", transformers.__version__)

# Veriyi yükle
with open("hazir_dataset.json", "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)
dataset = dataset.train_test_split(test_size=0.1)

# Ses dosyasını numpy dizisine çevir
def speech_file_to_array(batch):
    speech_array, sampling_rate = torchaudio.load(batch["path"])

    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        speech_array = resampler(speech_array)
        sampling_rate = 16000

    batch["speech"] = speech_array[0].numpy()
    batch["sampling_rate"] = sampling_rate
    return batch

# Tokenizer ve Model
processor = WhisperProcessor.from_pretrained("openai/whisper-medium", language="Turkish", task="transcribe")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="turkish", task="transcribe")

# Veriyi ses-diziye çevir
dataset = dataset.map(speech_file_to_array)

# Veriyi modele uygun hale getir
def prepare_dataset(batch):
    inputs = processor(
        batch["speech"],
        sampling_rate=batch["sampling_rate"],
        text=batch["transcription"],
        return_tensors="pt",
        padding="longest",
        max_length=448,
        truncation=True
    )
    batch["input_features"] = inputs["input_features"].squeeze(0)
    batch["labels"] = inputs["labels"].squeeze(0)
    return batch

# 🔧 BURASI EKLENMELİYDİ — SORUN BUNUN EKSİK OLMASIYDI
dataset = dataset.map(prepare_dataset)

# Data collator
def data_collator(features):
    input_features = [torch.tensor(f["input_features"]) for f in features]
    input_features_padded = torch.nn.utils.rnn.pad_sequence(input_features, batch_first=True)

    label_features = [torch.tensor(f["labels"]) for f in features]
    label_features_padded = torch.nn.utils.rnn.pad_sequence(label_features, batch_first=True, padding_value=-100)

    return {
        "input_features": input_features_padded,
        "labels": label_features_padded,
    }

# Eğitim ayarları
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    evaluation_strategy="epoch",
    logging_strategy="steps",
    logging_steps=10,
    save_strategy="epoch",
    num_train_epochs=3,
    gradient_checkpointing=True,
    fp16=False,
    report_to="none",
    remove_unused_columns=False,
)

# Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=processor,
    data_collator=data_collator,
)

# Eğitimi başlat
trainer.train()

# Modeli kaydet
model.save_pretrained("./final_model2")
processor.save_pretrained("./final_model2")
