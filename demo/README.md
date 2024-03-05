## How to use this demo?

1. Make sure you have atleast 16 GB GPU. (CPU also runs fine but it is very slow)
2. Run the cells, set HF_token and model_id
3. Set the system prompt. In this example I set the system prompt in alpaca_format
4. alpaca_format needs instruction and input. **Note : if you want to send instruction and input both, make sure they are separated with instruction ### input. The code will still work if you have not specified the ###**
5. If you are facing any issues in environment setup see the requirements.txt file in misc folder.
6. Also check the default parameters setup for inference like :
```
  max_new_tokens=2048,
  top_p=0.2,
  top_k=20,
  temperature=0.1,
  repetition_penalty=2.0,
  length_penalty=-0.5,
  num_beams=1
```
