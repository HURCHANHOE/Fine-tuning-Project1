from typing import List, Optional
from dataclasses import dataclass
# from torch.utils.data.dataset import Dataset
from filelock import FileLock
from ktnlp.common.nlpargs import TrainArguments
from transformers import PreTrainedTokenizer
import torch
import time
import os
from torch.utils.data import Dataset

def getDataSetLength(dataSet):
	totalLength = 0
	for i in dataSet:
		textLenth = 0
		textLenth = len(i)
		totalLength += textLenth
		
	return totalLength
	
class T5ClassificationDataSet_t2t_summarization(Dataset):
	def __init__(self,target_tokenizer,utterances,labels,utterance_max_length,label_max_length):
		self.utterances=utterances
		self.labels=labels
		tokenized_utterances=target_tokenizer.batch_encode_plus(utterances,padding="max_length",max_length=utterance_max_length,truncation=True,return_tensors='pt')

		self.source_ids=tokenized_utterances['input_ids']
		self.source_mask=tokenized_utterances['attention_mask']
		# tokenizer_labels=target_tokenizer.batch_encode_plus(labels,padding="longest",truncation=True,return_tensors='pt')
		tokenizer_labels = target_tokenizer.batch_encode_plus([sequence for sequence in labels], padding="max_length", max_length=label_max_length,
															  truncation=True, return_tensors='pt')
		self.target_ids=tokenizer_labels['input_ids']
		self.target_ids[self.target_ids[:, :] == target_tokenizer.pad_token_id] = -100
		self.target_mask=tokenizer_labels['attention_mask']
		
		self.total_token_count = 0
		tokens=target_tokenizer.batch_encode_plus(utterances,add_special_tokens=False,max_length=utterance_max_length,truncation=True)['input_ids']
		for token in tokens:
			self.total_token_count += len(token)
			
		self.total_length = getDataSetLength(utterances)
	def __len__(self):
		return len(self.source_ids)

	def __getitem__(self, i):
		return {'input_ids':self.source_ids[i],'attention_mask':self.source_mask[i], 'labels':self.target_ids[i],'decoder_attention_mask':self.target_mask[i]}
	
	def getTokenCount(self):
		return self.total_token_count

	def getDataLength(self):
		return self.total_length