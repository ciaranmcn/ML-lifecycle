from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from peft import get_peft_model, LoraConfig, TaskType
import os


def load_data(path):
    ds = load_dataset("json", data_files=path, split="train")
    ds = ds.map(lambda e: {"text": e["prompt"] + e["response"]})
    return ds

def tokenize_data(dataset, tokenizer) :
    def tokenize(example):
        return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)
    return dataset.map(tokenize)

def train_main(model_name: str, data_path: str):
    print("Loading tokenizer and model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)

    print("Preparing dataset...")
    dataset = load_data(data_path)
    dataset = tokenize_data(dataset, tokenizer)

    # Start Training here

    peft_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1
    )
    model = get_peft_model(model, peft_config)

    print("Setting up trainer...")

    training_args = TrainingArguments(
        output_dir="./model_output",
        overwrite_output_dir=True,
        num_train_epochs=True,
        num_device_train_batch_size=4,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        logging_steps=100,
        evaluation_strategy="no",
        fp16=True if torch.cuda.is_available() else False,
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=False
    )

    trainer = Trainer(
        model=model,
       args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    print("Training...")
    trainer.train()
    
    print("Saving model...")
    model.save_pretrained("./model_output")
    tokenizer.save_pretrained("./model_output")

    return "Training complete"