import json
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
import torch

# 1. 加载数据

def load_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f]
    return data

train_data = load_jsonl('domain_recognition_train.jsonl')
# 如有验证集，可类似加载
# val_data = load_jsonl('state_extraction_valid.jsonl')

dataset = Dataset.from_list(train_data)
# val_dataset = Dataset.from_list(val_data)

# 2. 加载分词器和模型
model_name = "Qwen/Qwen3-4B"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

# 建议用 4bit 加载节省显存
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    trust_remote_code=True,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    load_in_4bit=True
)

# 3. LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
    target_modules=["q_proj", "v_proj"]  # Qwen3 推荐
)

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, lora_config)

# 4. 数据预处理

def preprocess(example):
    prompt = example['prompt']
    output = example['output']
    text = f"{prompt}\nAnswer: {output}"
    return tokenizer(text, truncation=True, max_length=512, padding='max_length')

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)
# tokenized_val_dataset = val_dataset.map(preprocess, remove_columns=val_dataset.column_names)

# 5. 训练参数
training_args = TrainingArguments(
    output_dir='./qwen3_lora_output',
    per_device_train_batch_size=4,  # A10G可以支持更大批次
    gradient_accumulation_steps=2,  # 累积梯度，等效batch_size=8
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    learning_rate=2e-4,
    fp16=True,
    evaluation_strategy="no",
    save_total_limit=2,
    dataloader_pin_memory=True,  # A10G内存充足
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

# 7. 开始训练
if __name__ == "__main__":
    print(f"开始训练，样本数量: {len(tokenized_dataset)}")
    print(f"每个epoch步数: {len(tokenized_dataset) // training_args.per_device_train_batch_size}")
    print(f"有效批次大小: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
    trainer.train()
    model.save_pretrained('./qwen3_lora_output')
    tokenizer.save_pretrained('./qwen3_lora_output') 