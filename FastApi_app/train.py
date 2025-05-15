from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TraingingArguments, Trainer, DataCollatorForLanguageModeling
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
    return "Training complete"