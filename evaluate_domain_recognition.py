import json
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tqdm import tqdm
from collections import Counter
import argparse
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
import matplotlib.pyplot as plt

# ========== CONFIGURATION ==========
BERT_MODEL_PATH = 'bert-base-uncased'  # Use English BERT, or replace with your fine-tuned English model path
DOMAIN_LABELS = [
    'attraction', 'hotel', 'restaurant', 'taxi', 'train', 'hospital'
]  # Should match your training label set
PROMPT_FILE = 'bert_domain_eval_pairs.jsonl'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# ========== 1. Load Data ==========
def load_prompts(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        data = [json.loads(line) for line in f if line.strip()]
    samples = []
    for item in data:
        samples.append({
            'source': item['source'],
            'target': item['target'],
        })
    return samples

def load_texts_and_labels(filename):
    samples = load_prompts(filename)
    texts = [s['source'] for s in samples]
    labels = [s['target'] for s in samples]
    return texts, labels

# ========== 2. BERT Classifier ==========
class BertDomainClassifier:
    def __init__(self, model_path, label_list):
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path, num_labels=len(label_list))
        self.model.to(DEVICE)
        self.label2id = {l: i for i, l in enumerate(label_list)}
        self.id2label = {i: l for i, l in enumerate(label_list)}

    def predict(self, texts, batch_size=16):
        self.model.eval()
        preds = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc='BERT Predict'):
                batch = texts[i:i+batch_size]
                inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                outputs = self.model(**inputs)
                logits = outputs.logits
                batch_preds = torch.argmax(logits, dim=-1).cpu().tolist()
                preds.extend(batch_preds)
        return [self.id2label[i] for i in preds]

    def train(self, train_texts, train_labels, epochs=3, batch_size=16, lr=2e-5, save_path=None, eval_texts=None, eval_labels=None):
        from torch.utils.data import Dataset, DataLoader
        import numpy as np

        class TextDataset(Dataset):
            def __init__(self, texts, labels, label2id):
                self.texts = texts
                self.labels = [label2id[l] for l in labels]
            def __len__(self):
                return len(self.texts)
            def __getitem__(self, idx):
                return self.texts[idx], self.labels[idx]

        dataset = TextDataset(train_texts, train_labels, self.label2id)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        optimizer = AdamW(self.model.parameters(), lr=lr)
        total_steps = len(dataloader) * epochs
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
        loss_fn = torch.nn.CrossEntropyLoss()

        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        self.model.train()
        for epoch in range(epochs):
            total_loss = 0
            all_preds = []
            all_labels = []
            for batch_texts, batch_labels in tqdm(dataloader, desc=f'Epoch {epoch+1}/{epochs}'):
                inputs = self.tokenizer(list(batch_texts), padding=True, truncation=True, max_length=256, return_tensors='pt')
                inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                batch_labels = torch.tensor(batch_labels).to(DEVICE)
                outputs = self.model(**inputs)
                loss = loss_fn(outputs.logits, batch_labels)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()
                total_loss += loss.item()
                # 记录训练集预测
                preds = torch.argmax(outputs.logits, dim=-1).detach().cpu().tolist()
                all_preds.extend(preds)
                all_labels.extend(batch_labels.detach().cpu().tolist())
            avg_loss = total_loss / len(dataloader)
            train_acc = accuracy_score(all_labels, all_preds)
            history['train_loss'].append(avg_loss)
            history['train_acc'].append(train_acc)
            print(f'Epoch {epoch+1} 平均loss: {avg_loss:.4f} 训练集准确率: {train_acc:.4f}')
            # 验证集评估
            val_loss, val_acc = None, None
            if eval_texts and eval_labels:
                self.model.eval()
                val_preds = []
                val_losses = []
                with torch.no_grad():
                    for i in range(0, len(eval_texts), batch_size):
                        batch = eval_texts[i:i+batch_size]
                        labels = eval_labels[i:i+batch_size]
                        inputs = self.tokenizer(batch, padding=True, truncation=True, max_length=256, return_tensors='pt')
                        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
                        label_ids = torch.tensor([self.label2id[l] for l in labels]).to(DEVICE)
                        outputs = self.model(**inputs)
                        loss = loss_fn(outputs.logits, label_ids)
                        val_losses.append(loss.item())
                        preds = torch.argmax(outputs.logits, dim=-1).cpu().tolist()
                        val_preds.extend(preds)
                val_pred_labels = [self.id2label[i] for i in val_preds]
                val_acc = accuracy_score(eval_labels, val_pred_labels)
                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
                print(f'Epoch {epoch+1} 验证集loss: {val_loss:.4f} 验证集准确率: {val_acc:.4f}')
                self.model.train()
            else:
                history['val_loss'].append(None)
                history['val_acc'].append(None)
        if save_path:
            self.model.save_pretrained(save_path)
            self.tokenizer.save_pretrained(save_path)
            print(f'模型已保存到: {save_path}')
        # 画曲线
        epochs_range = range(1, epochs+1)
        plt.figure(figsize=(10,5))
        plt.plot(epochs_range, history['train_loss'], label='Train Loss')
        if all(v is not None for v in history['val_loss']):
            plt.plot(epochs_range, history['val_loss'], label='Val Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Loss Curve')
        plt.savefig('train_val_loss_curve.png')
        plt.close()
        plt.figure(figsize=(10,5))
        plt.plot(epochs_range, history['train_acc'], label='Train Acc')
        if all(v is not None for v in history['val_acc']):
            plt.plot(epochs_range, history['val_acc'], label='Val Acc')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.title('Accuracy Curve')
        plt.savefig('train_val_acc_curve.png')
        plt.close()
        print('训练/验证集loss和accuracy曲线已保存为train_val_loss_curve.png和train_val_acc_curve.png')

# ========== 3. LLM Prompt Classification ==========

# ========== 4. Evaluation ==========
def evaluate(gold, pred, method_name):
    acc = accuracy_score(gold, pred)
    print(f'[{method_name}] Accuracy: {acc:.4f}')
    print(classification_report(gold, pred, labels=DOMAIN_LABELS, digits=3))
    print('Confusion Matrix:')
    print(confusion_matrix(gold, pred, labels=DOMAIN_LABELS))

# ========== 5. Main ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'eval'], default='eval', help='train or eval')
    parser.add_argument('--train_file', type=str, default='bert_domain_train_pairs.jsonl', help='训练集文件')
    parser.add_argument('--eval_file', type=str, default='bert_domain_eval_pairs.jsonl', help='评估集文件')
    parser.add_argument('--epochs', type=int, default=3, help='训练轮数')
    parser.add_argument('--save_path', type=str, default='bert_finetuned', help='模型保存路径')
    parser.add_argument('--model_path', type=str, default=BERT_MODEL_PATH, help='BERT模型路径')
    args = parser.parse_args()

    if args.mode == 'train':
        print('==== BERT Classifier Training ===')
        train_texts, train_labels = load_texts_and_labels(args.train_file)
        eval_texts, eval_labels = load_texts_and_labels(args.eval_file)
        # 打印训练集domain分布
        print('训练集 domain 分布：')
        for domain, count in Counter(train_labels).items():
            print(f'{domain}: {count}')
        print('-' * 40)
        # 打印验证集domain分布
        print('验证集 domain 分布：')
        for domain, count in Counter(eval_labels).items():
            print(f'{domain}: {count}')
        print('-' * 40)
        bert_clf = BertDomainClassifier(args.model_path, DOMAIN_LABELS)
        bert_clf.train(train_texts, train_labels, epochs=args.epochs, save_path=args.save_path, eval_texts=eval_texts, eval_labels=eval_labels)
    elif args.mode == 'eval':
        print('==== BERT Classifier Evaluation ===')
        eval_texts, eval_labels = load_texts_and_labels(args.eval_file)
        # domain分布
        domain_counter = Counter(eval_labels)
        print("评估数据 domain 分布：")
        for domain, count in domain_counter.items():
            print(f"{domain}: {count}")
        print("-" * 40)
        bert_clf = BertDomainClassifier(args.model_path, DOMAIN_LABELS)
        bert_preds = bert_clf.predict(eval_texts)
        evaluate(eval_labels, bert_preds, 'BERT')
    else:
        raise ValueError('mode必须是train或eval')

if __name__ == '__main__':
    main() 