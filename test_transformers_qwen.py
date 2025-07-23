import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = 'Qwen/Qwen3-4B'  # 可改为本地路径

def main():
    print('加载模型和分词器...')
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    print(f'模型已加载到: {device}')

    while True:
        text = input('\n请输入测试内容（输入exit退出）：')
        if text.strip().lower() == 'exit':
            break
        inputs = tokenizer(text, return_tensors='pt').to(device)
        print('正在生成回复...')
        outputs = model.generate(**inputs, max_new_tokens=64, do_sample=True, top_p=0.8)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f'模型回复: {response}')

if __name__ == '__main__':
    main() 