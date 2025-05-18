from datasets import load_dataset
import json
import os 

def format_instructions(example):
    return {
        "prompt": f"### Instruciton: \n{example['question']}\n\n### Response:",
        "response": example["response"]
    }

def save_to_json1(dataset, output_path):
    with open(output_path, "w") as f:
        for example in dataset:
            json.dump(example, f)
            f.write("\n")
    print(f"Saved {len(dataset)} entries to {output_path}")

def preprocess_main(dataset, sample_size):
    print(f"Loading {dataset} dataset with {sample_size} samples...")
    ds = load_dataset(dataset, split=f"train[:{sample_size}]")

    print("Formatting examples...")
    formatted = ds.map(format_instructions)

    print("Making formatted data file")
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    output_dir = f"datasets/{dataset.replace('/', '_')}"
    os.makedirs(output_dir, exist_ok=True)
    output_path = f"{output_dir}/preprocessed_{timestamp}.jsonl"

    save_to_json1(formatted, output_path)

    return output_path
