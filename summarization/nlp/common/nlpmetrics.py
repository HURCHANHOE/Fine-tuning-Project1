import torch
import logging
from rouge import Rouge
from time import time

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

import numpy as np

rouge_scorer = Rouge()
class compute_metrics : 
	def __init__(self, fast_tokenizer):
		self.fast_tokenizer = fast_tokenizer
		
	def compute_metrics_rouge(self, p):
		preds, labels = p

		# TODO 전체 Validation 결과 활용할 때 주석 필요
		# preds, labels = preds[:200], labels[:200]

		logger.info("fast tokenizer start")
		tokenized_preds = [self.fast_tokenizer.convert_ids_to_tokens(pred, skip_special_tokens=True) for pred in preds]
		tokenized_labels = [self.fast_tokenizer.convert_ids_to_tokens(label, skip_special_tokens=True) for label in labels]
		logger.info("fast tokenizer end")

		tokenized_preds = [' '.join(pred) for pred in tokenized_preds]
		tokenized_labels = [' '.join(label) for label in tokenized_labels]
		
		try :
			
			if tokenized_preds == '' :
				rouge_val = {'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0},
							'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0},
							'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
			else :
				rouge_val = rouge_scorer.get_scores(tokenized_preds, tokenized_labels, avg=True)
		except Exception as e :
			rouge_val = {'rouge-1': {'r': 0.0, 'p': 0.0, 'f': 0.0},
						'rouge-2': {'r': 0.0, 'p': 0.0, 'f': 0.0},
						'rouge-l': {'r': 0.0, 'p': 0.0, 'f': 0.0}}
			logger.info(e)

		rouge_val_f1 = {score_type: scores['f'] for score_type, scores in rouge_val.items()}
		rouge_val_recall = {score_type: scores['r'] for score_type, scores in rouge_val.items()}
		rouge_val_precision = {score_type: scores['p'] for score_type, scores in rouge_val.items()}

		return {"f1_rouge": rouge_val_f1['rouge-l'], "recall_rouge": rouge_val_recall['rouge-l'], "precision_rouge": rouge_val_precision['rouge-l']}
