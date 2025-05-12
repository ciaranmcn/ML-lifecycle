from datasets import load_dataset
import json
import os 

OUTPUT_PATH = "datasets/openorca_formatted.jsonl"
SAMPLE_SIZE = 5000

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

def main():
    print("Loading OpenOrca dataset...")
    ds = load_dataset("Open-Orca/OpenOrca", split=f"train[:{SAMPLE_SIZE}]")

    print("Formatting examples...")
    formatted = ds.map(format_instructions)

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    save_to_json1(formatted, OUTPUT_PATH)

if __name__ == "__main__":
    main()