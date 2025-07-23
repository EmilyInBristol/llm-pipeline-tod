# #!/bin/bash

# # 训练BERT模型
# python3 evaluate_domain_recognition.py \
#   --mode train \
#   --train_file bert_domain_train_pairs.jsonl \
#   --eval_file bert_domain_eval_pairs.jsonl \
#   --epochs 3 \
#   --save_path bert_finetuned

# # 评估BERT模型
# python3 evaluate_domain_recognition.py \
#   --mode eval \
#   --eval_file bert_domain_eval_pairs.jsonl \
#   --model_path bert_finetuned 

python3 evaluate_domain_recognition.py \
  --mode eval \
  --eval_file bert_domain_eval_pairs.jsonl \

