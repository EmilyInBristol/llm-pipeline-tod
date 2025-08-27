import argparse
import json
import os
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch
# 1. Load data

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_file', type=str, required=True, help='Path to training data file')
    args = parser.parse_args()

    train_data = load_jsonl(args.train_file)
    dataset = Dataset.from_list(train_data)

    # 2. Load tokenizer and model
    model_name = "Qwen/Qwen3-4B"
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    # Load model in 4bit to save memory
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        trust_remote_code=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        load_in_4bit=True
    )

    # 3. LoRA config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "v_proj"]  # Recommended for Qwen3
    )

    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, lora_config)

    # 4. Data preprocessing
    def preprocess(example):
        prompt = example['prompt']
        output = example['output']
        text = f"{prompt}\n{output}"
        return tokenizer(text, truncation=True, max_length=512, padding='max_length')

    tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # 5. Training arguments
    train_file_base = os.path.splitext(os.path.basename(args.train_file))[0]
    output_dir = os.path.join('./qwen3_lora_output', train_file_base)
    os.makedirs(output_dir, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=4,  # Larger batch size supported on A10G
        gradient_accumulation_steps=2,  # Effective batch_size=8
        num_train_epochs=3,
        logging_steps=10,
        save_steps=100,
        learning_rate=2e-4,
        fp16=True,
        # evaluation_strategy="no",
        save_total_limit=2,
        dataloader_pin_memory=True,  # Sufficient memory on A10G
        remove_unused_columns=False,
        report_to=None,
    )

    # 6. Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    # 7. Start training
    print(f"Start training, number of samples: {len(tokenized_dataset)}")
    print(f"Steps per epoch: {len(tokenized_dataset) // training_args.per_device_train_batch_size}")
    print(f"Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    print(f"Model output directory: {output_dir}")
    trainer.train()
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir) 