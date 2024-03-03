import argparse
from unsloth import FastLanguageModel
import torch


def parse_arguments():
    parser = argparse.ArgumentParser(description="Translate text to Hindi using a pretrained language model.")
    parser.add_argument("input_file", type=str, help="Path to the input file containing text to translate.")
    parser.add_argument("output_file", type=str, help="Path to the output file to save translated text.")
    return parser.parse_args()


def load_model():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="lora_model",  
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=False,
    )
    FastLanguageModel.for_inference(model)  
    return model, tokenizer


def generate_translations(input_file, output_file, model, tokenizer):
    with open(input_file, encoding='utf-8') as file:
        data = file.readlines()

    data_list = []
    for item in data:
        data_list.append(alpaca_prompt.format("Translate this to Hindi", item, ""))

    inputs = tokenizer(
        data_list, padding=True, return_tensors="pt").to("cuda")

    outputs = model.generate(**inputs, max_new_tokens=256, use_cache=True)
    decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    with open(output_file, 'w+', encoding='utf-8') as file:
        for i, output in enumerate(decoded_outputs, start=0):
            print(f"{output}", file=file)


if __name__ == "__main__":
    args = parse_arguments()
    model, tokenizer = load_model()
    generate_translations(args.input_file, args.output_file, model, tokenizer)
