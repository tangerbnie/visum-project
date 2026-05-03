"""
ViSum - QLoRA Fine-tune BARTpho trên 144K mẫu VietNews (5 Epochs)
GPU: RTX 3090 24GB
Author: OrdinaryAI
"""

from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments, Seq2SeqTrainer,
    DataCollatorForSeq2Seq, BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
import torch
import os

# ============================================
# 1. LOAD DATASET (144K MẪU)
# ============================================
print("=" * 60)
print("BƯỚC 1: Load dataset VietNews (144K mẫu)")
print("=" * 60)
dataset = load_dataset("harouzie/vietnews")
train_data = dataset['train']        # 99.134 mẫu
val_data = dataset['validation']     # 22.184 mẫu
print(f"Train: {len(train_data)} | Val: {len(val_data)}")

# ============================================
# 2. TOKENIZE
# ============================================
print("\n" + "=" * 60)
print("BƯỚC 2: Tokenize")
print("=" * 60)
MODEL_NAME = "vinai/bartpho-syllable"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def preprocess(examples):
    inputs = tokenizer(examples['article'], max_length=512, truncation=True)
    labels = tokenizer(examples['abstract'], max_length=128, truncation=True)
    inputs['labels'] = labels['input_ids']
    return inputs

tokenized_train = train_data.map(
    preprocess, batched=True, 
    remove_columns=train_data.column_names
)
tokenized_val = val_data.map(
    preprocess, batched=True, 
    remove_columns=val_data.column_names
)
print(f"Tokenized: {len(tokenized_train)} train | {len(tokenized_val)} val")

# ============================================
# 3. QLoRA CONFIG + LOAD MODEL
# ============================================
print("\n" + "=" * 60)
print("BƯỚC 3: Load Model với QLoRA (4-bit)")
print("=" * 60)
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
)
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],
    lora_dropout=0.1, bias="none",
    task_type=TaskType.SEQ_2_SEQ_LM,
)
model = get_peft_model(model, lora_config)
print("Trainable parameters: ", end="")
model.print_trainable_parameters()

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# ============================================
# 4. TRAINING (5 EPOCHS)
# ============================================
print("\n" + "=" * 60)
print("BƯỚC 4: Train 5 Epochs")
print("=" * 60)

training_args = Seq2SeqTrainingArguments(
    output_dir="./visum-qlora-5epochs",
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_steps=500,
    per_device_train_batch_size=8,
    gradient_accumulation_steps=2,
    per_device_eval_batch_size=8,
    learning_rate=2e-4,
    num_train_epochs=5,                     # ← 5 EPOCHS
    predict_with_generate=True,
    generation_max_length=150,
    fp16=True,
    save_total_limit=5,
    report_to="none",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    processing_class=tokenizer,
    data_collator=data_collator,
)

print("🚀 Bắt đầu train 144K mẫu - 5 epochs...")
print(f"⏱️  Dự kiến: ~24 tiếng (mỗi epoch ~4.8 tiếng)")

trainer.train()

# ============================================
# 5. LƯU MODEL
# ============================================
print("\n" + "=" * 60)
print("BƯỚC 5: Lưu Model")
print("=" * 60)
model.save_pretrained("./visum-qlora-5epochs")
tokenizer.save_pretrained("./visum-qlora-5epochs")
print("✅ Done! Model saved to ./visum-qlora-5epochs")

# ============================================
# 6. THÔNG TIN HUẤN LUYỆN
# ============================================
print("\n" + "=" * 60)
print("THÔNG TIN HUẤN LUYỆN")
print("=" * 60)
print(f"  Model gốc:       {MODEL_NAME}")
print(f"  Dataset:         harouzie/vietnews (144K mẫu)")
print(f"  Phương pháp:     QLoRA (4-bit) + LoRA (r=16, alpha=32)")
print(f"  Epochs:          5")
print(f"  Batch size:      8 × gradient_accumulation 2 = 16")
print(f"  Learning rate:   2e-4")
print(f"  GPU:             NVIDIA GeForce RTX 3090 24GB")
print(f"  Thư mục output:  ./visum-qlora-5epochs")