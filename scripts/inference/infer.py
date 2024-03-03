import argparse
from unsloth import FastLanguageModel
import torch

def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate text to Hindi using a pretrained language model.")
    parser.add_argument("--instruction", type=str, help="Instruction for translation")
    parser.add_argument("--input_text", type=str, help="Input text for translation")
    parser.add_argument("output_file", type=str, help="Path to the output file to save translated text.")
    return parser.parse_args()

def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  # YOUR MODEL YOU USED FOR TRAINING
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)  # Enable native 2x faster inference
    return model, tokenizer

def translate_single_instruction(instruction, input_text, output_file, model, tokenizer):
    alpaca_prompt = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.

### Instruction:
{}

### Input:
{}

### Response:
{}""".format(instruction, input_text, "")

    inputs = tokenizer(
        [alpaca_prompt], padding=True, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    with open(output_file, 'w+', encoding='utf-8') as file:
        print(decoded_outputs[0], file=file)

if __name__ == "__main__":
    args = parse_arguments()
    model, tokenizer = load_model()
    translate_single_instruction(args.instruction, args.input_text, args.output_file, model, tokenizer)
