import argparse
from unsloth import FastLanguageModel
import torch
from tqdm import tqdm

def load_model(model_name, max_seq_length=2048, dtype=None, load_in_4bit=False, token=None):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=dtype,
        load_in_4bit=load_in_4bit,
        token=token
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer

def translate_text(data, model, tokenizer, output_file, infer_output_file, test_string):
    template = "Instruction:\n{instruction}\n\nResponse:\n{response}"
    template_infer = "Instruction:\n{instruction}\n\nResponse:\n"

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches)):
            batch_data = data[batch_idx * batch_size: (batch_idx + 1) * batch_size]
            data_list = []
            infer_list = []
            for item in batch_data:
                instruction = f"{test_string}\n\nTranslate the below sentence from English to Hindi in devanagari script.\n{item}"
                data_list.append(template.format(instruction=instruction, response=""))
                infer_list.append(template_infer.format(instruction=instruction))

            inputs = tokenizer(
                data_list,
                padding=True,
                return_tensors="pt",
                truncation=True
            )

            inputs = inputs.to("cuda")

            outputs = model.generate(input_ids=inputs.input_ids, max_length=256, use_cache=True)

            decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            with open(output_file, 'a', encoding='utf-8') as file, open(infer_output_file, 'a', encoding='utf-8') as lfile:
                for i, output in enumerate(decoded_outputs, start=0):
                    print(output.replace(infer_list[i], '').replace('\n', '').strip(), file=lfile)
                    print(output, file=file)

def main(args):
    model, tokenizer = load_model(args.model_name, token=args.token)

    with open(args.input_file, encoding='utf-8') as file:
        data = file.readlines()

    batch_size = 1
    num_batches = (len(data) + batch_size - 1) // batch_size

    translate_text(data, model, tokenizer, args.output_file, args.infer_output_file, args.test_string)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate text from English to Hindi using a pre-trained language model.")
    parser.add_argument("--model_name", type=str, required=True, help="Name of the pre-trained language model")
    parser.add_argument("--token", type=str, required=True, help="Token for Hugging Face Hub login")
    parser.add_argument("--input_file", type=str, required=True, help="Path to the input text file")
    parser.add_argument("--output_file", type=str, required=True, help="Path to the output text file")
    parser.add_argument("--infer_output_file", type=str, required=True, help="Path to the inference output text file")
    parser.add_argument("--test_string", type=str, required=True, help="String for testing")
    args = parser.parse_args()
    main(args)
