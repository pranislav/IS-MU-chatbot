import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding

# Load dataset (replace with your dataset path or name)
dataset = load_dataset("json", data_files="raw_QA.json")

def format_qa(example):
    example["input_text"] = f"Question: {example['question']}"
    example["target_text"] = f"Answer: {example['answer']}"
    return example

dataset = dataset.map(format_qa)

dataset = dataset["train"].train_test_split(test_size=0.2)  # Train-validation split

# Load distilGPT2 tokenizer and model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")
tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
tokenizer.pad_token = tokenizer.eos_token


# Tokenize dataset
def preprocess_function(example):
    return tokenizer(example["input_text"], truncation=True, padding=True, max_length=256)


tokenized_datasets = dataset.map(preprocess_function, batched=True)

# Define QLoRA config
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices
    lora_alpha=16,
    # target_modules=["query", "value"],  # Target attention layers
    lora_dropout=0.1,
    bias="none"
)

# Wrap model with LoRA adapter
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=3,
    logging_dir="./logs",
    logging_steps=10,
    load_best_model_at_end=True,
    metric_for_best_model="loss",
    greater_is_better=False,
    save_on_each_node=False,
    fp16=True if torch.cuda.is_available() else False,
    report_to="none"
)


# Data collator for dynamic padding
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# Trainer setup
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# Fine-tune the model
trainer.train()

# Save final model
model.save_pretrained("./fine_tuned_tinybert")
tokenizer.save_pretrained("./fine_tuned_tinybert")
