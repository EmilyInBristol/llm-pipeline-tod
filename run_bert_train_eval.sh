# #!/bin/bash

# # Train BERT model
# python3 evaluate_domain_recognition.py \
#   --mode train \
#   --train_file bert_domain_train_pairs.jsonl \
#   --eval_file bert_domain_eval_pairs.jsonl \
#   --epochs 3 \
#   --save_path bert_finetuned

# # Evaluate BERT model
# python3 evaluate_domain_recognition.py \
#   --mode eval \
#   --eval_file bert_domain_eval_pairs.jsonl \
#   --model_path bert_finetuned 

python3 evaluate_domain_recognition.py \
  --mode eval \
  --eval_file bert_domain_eval_pairs.jsonl \

