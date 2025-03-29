import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer, DataCollatorWithPadding, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,  # Use 4-bit quantization
    bnb_4bit_compute_dtype=torch.float16,  # Use float16 for computations
    bnb_4bit_use_double_quant=True,  # Further reduce memory
)

# It is strongly recommended to train Gemma3 models with the
# `eager` attention implementation instead of `sdpa`. Use `eager` with
# `AutoModelForCausalLM.from_pretrained('<path-to-checkpoint>', attn_implementation='eager')`.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-4b-it", attn_implementation='eager', quantization_config=bnb_config)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-4b-it")


# Load dataset (replace with your dataset path or name)
dataset = load_dataset("json", data_files="dataset/raw_QA.json")

def format_qa(example):
    messages = [
        {"role": "user", "content": example["question"]},
        {"role": "assistant", "content": example["answer"]},
    ]
    example["formatted_text"] = tokenizer.apply_chat_template(messages, tokenize=False)
    return example

# Tokenize dataset
def tokenize_function(example):
    # Tokenize the formatted text
    encoding = tokenizer(example["formatted_text"], truncation=True, padding=True, max_length=512) # TODO: remove max length when dynamic padding implemented

    # Create labels by shifting the input_ids (next token prediction task)
    encoding["labels"] = encoding["input_ids"].copy()
    
    # remove padding tokens from labels
    if tokenizer.pad_token_id is not None:
        encoding["labels"] = [
            label if label != tokenizer.pad_token_id else -100
            for label in encoding["labels"]
        ]

    return encoding

dataset = dataset.map(format_qa)
dataset = dataset["train"].train_test_split(test_size=0.2)  # Train-validation split
tokenized_datasets = dataset.map(tokenize_function, batched=True, remove_columns=["formatted_text"])

# Define QLoRA config
lora_config = LoraConfig(
    r=8,  # Rank of the LoRA matrices
    lora_alpha=16,
    target_modules=["q_proj", "k_proj", "v_proj", "out_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM",
)

# Wrap model with LoRA adapter
model = get_peft_model(model, lora_config)

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    save_total_limit=2,
    gradient_accumulation_steps=4,  # Accumulate over 4 small batches
    per_device_train_batch_size=1,  # Reduce per-GPU batch size
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
model.save_pretrained("./models/IS-tuned_gemma3-4b-it")
tokenizer.save_pretrained("./models/IS-tuned_gemma3-4b-it")
