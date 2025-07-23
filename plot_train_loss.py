import re
import matplotlib.pyplot as plt

losses = []
epochs = []

# 读取nohup.out文件
with open('./nohup.out', 'r', encoding='utf-8') as f:
    for line in f:
        # 匹配包含loss和epoch的行
        match = re.search(r"'loss': ([\d\.]+).*'epoch': ([\d\.]+)", line)
        if match:
            loss = float(match.group(1))
            epoch = float(match.group(2))
            losses.append(loss)
            epochs.append(epoch)

# 绘图
plt.figure(figsize=(8, 5))
plt.plot(epochs, losses, marker='o')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Curve')
plt.grid(True)
plt.tight_layout()
plt.show() 