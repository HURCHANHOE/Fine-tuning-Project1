# pip3 install -r requirements.txt

echo ===== FINETUNNING =====
MODEL_PATH=paust/pko-t5-small
# TITLE=t5-small-finetuned
# DATA=data/개인및관계_1000.csv
# config_file_path='{"params": [{"param_name": "gradient_accumulation_steps","param_value": 8,"param_describe": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass. (int)"},{"param_name": "max_seq_length","param_value": 512,"param_describe": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. (int)​"},{"param_name": "split_ratio","param_value": [0.90,0.10,0.20],"param_describe": "Train-Validation Split ratio"},{"param_name": "epochs","param_value": 30,"param_describe": "Total number of training epochs to perform. (int)​"},{"param_name": "learning_rate","param_value": 5.00E-04,"param_describe": "The initial learning rate for the optimizer. (float)​"},{"param_name": "early_stopping_flag","param_value": 1,"param_describe": "early stopping apply flag(0 = False, 1 = True)​"},{"param_name": "early_stopping_patience","param_value": 3,"param_describe": "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls.(int, scope : 3~5)"}]}'
# bucket_name='t5-small'

OUTPUT_DIR=/result
mkdir $OUTPUT_DIR

echo ===== current OUTPUT_DIR is $OUTPUT_DIR =====
echo ===== MODEL_PATH is $MODEL_PATH =====

# gpu가 v100인경우 bf16 tf32 True 옵션 작동안함
python3 summarization_train_pipeline.py