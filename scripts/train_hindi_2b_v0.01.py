import argparse
from unsloth import FastLanguageModel
import torch
from huggingface_hub import login
from datasets import load_dataset, concatenate_datasets
from trl import SFTTrainer
from transformers import TrainingArguments

def load_model(model_name, max_seq_length=2048, dtype=None, load_in_4bit=False):
    login(token=args.token)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha=16,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing=True,
        random_state=3407,
        use_rslora=False,
        loftq_config=None,
    )

    return model, tokenizer

def format_prompts(examples):
    alpaca_prompt = """Below is an instruction that describes a task, paired with a optional input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}"""
    
    texts = []
    for instruction, input, output in zip(examples["instruction"], examples["input"], examples["output"]):
        text = alpaca_prompt.format(instruction, input, output) + tokenizer.eos_token
        texts.append(text)
    return {"text": texts}

def load_datasets():
    dataset1 = load_dataset("yahma/alpaca-cleaned", split="train")
    dataset1 = dataset1.map(format_prompts, batched=True)

    dataset2 = load_dataset("BhabhaAI/indic-instruct-data-v0.1-filtered", "dolly", split="hi")
    dataset2 = dataset2.rename_column("context", "input")
    dataset2 = dataset2.rename_column("response", "output")
    dataset2 = dataset2.map(format_prompts, batched=True)

    dataset3 = load_dataset("BhabhaAI/indic-instruct-data-v0.1-filtered", "dolly", split="en")
    dataset3 = dataset3.rename_column("context", "input")
    dataset3 = dataset3.rename_column("response", "output")
    dataset3 = dataset3.map(format_prompts, batched=True)

    dataset4 = load_dataset("BhabhaAI/indic-instruct-data-v0.1-filtered", "nmt-seed", split="hi")
    dataset4 = dataset4.rename_column("input_text", "input")
    dataset4 = dataset4.rename_column("output_text", "output")
    dataset4 = dataset4.map(add_instruction, batched=True)

    concatenated_dataset = concatenate_datasets([dataset1, dataset2, dataset3, dataset4])
    return concatenated_dataset

def add_instruction(example):
    example["instruction"] = "Translate this sentence to Hindi"
    return example

def filter_example(example):
    return len(example['text']) <= 2048

def main(args):
    model, tokenizer = load_model(args.model_name)

    concatenated_dataset = load_datasets()
    filtered_dataset = concatenated_dataset.filter(filter_example)

    train_size = 0.9
    train_dataset = filtered_dataset.train_test_split(test_size=1 - train_size)["train"]
    validation_dataset = filtered_dataset.train_test_split(test_size=1 - train_size)["test"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=2,
        packing=True,
        args=TrainingArguments(
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            num_train_epochs=2,
            learning_rate=2e-4,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=1,
            optim="paged_adamw_32bit",
            weight_decay=0.01,
            lr_scheduler_type="linear",
            seed=3407,
            output_dir="outputs",
        ),
    )

    trainer_stats = trainer.train()
    model.save_pretrained("lora_model")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a model for instructional text translation.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained language model")
    parser.add_argument("--token", type=str, required=True, help="Token for Hugging Face Hub login")
    parser.add_argument("--max_seq_length", type=int, default=2048, help="Maximum sequence length")
    args = parser.parse_args()
    main(args)
