# gemma-experimentation
Experimentation on google's gemma model

## Gradio Inference Demos 
| Model | Link |
| --- | ---- |
| Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa | [Link](https://github.com/vakyansh/gemma-experimentation/blob/main/demo/unsloth_infer_2b.ipynb) |
| Telugu-LLM-Labs/Indic-gemma-7b-finetuned-sft-Navarasa | [Link](https://github.com/vakyansh/gemma-experimentation/blob/main/demo/unsloth_infer_7b_4bit.ipynb) |


## Datasets used for hi-gemma2b-ft-lora-v0.01
BhabhaAI/indic-instruct-data-v0.1-filtered

yahma/alpaca-cleaned

## Benchmarking

### Translation

| Model            | 0-shot BLEU |
|------------------|----------------|
| gemma2b          | 2.84           |
| gemma2b-it       | 5.21           |
| Telugu-LLM-Labs/Indic-gemma-2b-finetuned-sft-Navarasa | 6.22 |
| hi-gemma2b-ft-lora-v0.01| 19.59          |




## Learnings:
06/03/2024: Added inferene client using gradio. Nice looking UI

02/03/2024: Use packing=True as suggested by Ravi. Avoids nan and inf gradients during training. Also decreases training time.

Will update more learnings as I keep one experimenting.
