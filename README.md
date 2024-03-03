# gemma-experimentation
Experimentation on google's gemma model

## Datasets used for gemma2b-hi-v0.01
BhabhaAI/indic-instruct-data-v0.1-filtered

yahma/alpaca-cleaned

## Benchmarking

### Translation

| Model            | 0-shot BLEU |
|------------------|----------------|
| gemma2b          | 2.84           |
| gemma2b-it       | 5.21           |
| gemmat2b-hi-v0.01| 19.59          |




## Learnings:

02/03/2024: Use packing=True as suggested by Ravi. Avoids nan and inf gradients during training. Also decreases training time.

Will update more learnings as I keep one experimenting.
