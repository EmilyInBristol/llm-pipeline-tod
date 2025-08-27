import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'Qwen/Qwen3-4B'  # Can be changed to local path

def main():
    print('Loading model and tokenizer...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'Model loaded to: {device}')

    while True:
        text = input('\nEnter test content (enter exit to quit): ')
        if text.strip().lower() == 'exit':
            break
        inputs = tokenizer(text, return_tensors='pt').to(device)
        print('Generating response...')
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.8)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'Model response: {response}')

if __name__ == '__main__':
    main() 