import os, sys
import pickle
import glob
import torch
import json
from transformers import T5Config,T5Tokenizer, T5ForConditionalGeneration
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))
from nlp.common.common import set_service_gpu 
from nlp.common.common import extractall_filetunning_model
from nlp.common.common import getDataSetTokenCount 
from interface.serviceInterface import servicePipelineInterface
from transformers.adapters import AdapterConfig, ConfigUnion
from tokenizers import AddedToken
from time import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)
class SummarizationServicePipeline(servicePipelineInterface):
	# model_static = None
	# tokenizer_static = None
	# gpu 세팅 및 학습정도 세팅
	def __init__(self):

		self.pelt_task_name="kt_peft"


	# finetunning 모델 로드 및 Tokenizer 로드
	def load_model(self, finetunning_path, base_path):
		# gpu/cpu 셋팅
		self.device=torch.device(set_service_gpu())
		logger.info(self.device)
		self.base_path = base_path
		# 압축해제 파일 삭제
		print('start extractall_filetunning_model')
		path = extractall_filetunning_model(finetunning_path)
		print('end extractall_filetunning_model')
		self.finetunning_path = path
		learn_info_file_path=os.path.join(self.finetunning_path,'learn_info.pkl') 
		# 학습 정보 로드
		with open(learn_info_file_path,'rb') as learnInfo:
				learn_info=pickle.load(learnInfo)
		logger.info(base_path)
		if base_path == None:
			# base_path 로드
			with open(self.finetunning_path+"/config.json",'r') as f:
				config=json.load(f)
			self.base_path = config["_name_or_path"]
			logger.info("base_path")
			logger.info(self.base_path)
		else:
			self.base_path = base_path
			logger.info("base_path")
			logger.info(self.base_path)

		# 학습에서 저장된 정보를 inference에 적용
		self.max_seq_length=learn_info['max_seq_length']
		# self.label_idx=learn_info['intent_list']
		# self.softmax_layer_applied=False if not learn_info.get('softmax_flag') else learn_info.get('softmax_flag')
		# 1. Tokenizer 로드
		# train pipeline에서 Tokenizer를 모델명으로 로드했으므로 동일한 방식으로 로드해야한다.
		add_tokens=[]
		CLS_TOKEN="<s>"
		SEP_TOKEN="</s>"
		tokenizer = T5Tokenizer.from_pretrained(self.base_path, extra_ids=0, cls_token=CLS_TOKEN,sep_token=SEP_TOKEN)
        
		tokeBytes_path = os.path.join(self.finetunning_path, 'bytes.t')
		BYTE_TOKENS = {i: byte[:-1] for i, byte in enumerate(open(tokeBytes_path).readlines())}
        
		add_tokens.extend([AddedToken(BYTE_TOKENS[t]) for t in BYTE_TOKENS])
		tokenizer.add_special_tokens({"additional_special_tokens": add_tokens})
  
		# Load adapter config
		with open(self.finetunning_path + '/peft_config.json', 'r') as f:
				peft_config_dict = json.load(f)

		peft_config = ConfigUnion.from_dict(peft_config_dict)
        
		config = T5Config.from_pretrained(self.finetunning_path)
		config.vocab_size = len(tokenizer)
		
		# model = T5ClassificationAdapter(self.base_path, config=config, pelt_config=pelt_config, tokenizer=tokenizer,
		# 											pelt_task_name=self.pelt_task_name, mode='test')
		model = T5ForConditionalGeneration.from_pretrained(self.base_path, config=config)
		model.set_active_adapters(self.pelt_task_name)
		model.resize_token_embeddings(len(tokenizer))
        
		# adapter and classifier model parameter load
		raw_state_dict = model.state_dict()
		freeze_state_dict = {}
		for key, value in raw_state_dict.items():
				if self.pelt_task_name not in key:
						freeze_state_dict[key] = value
		freeze_state_dict.update(torch.load(os.path.join(self.finetunning_path, 'peft_model.pt')))

		model.load_state_dict(freeze_state_dict)
		model.to(self.device)
		model.eval()

		## static 변수
		# ClassificationServicePipeline.tokenizer_static = tokenizer
		# ClassificationServicePipeline.model_static = model

		return model, tokenizer

	# 입력받은 문자에 대한 의도 
	def inference(self, model, tokenizer, sentence):

		# 추론 서비스 토큰 초기화 
		input_token_count = 0
		total_token_count = 0

		# 추론 서비스 result를 list->dict 변경 
		inference_result_list=[]
		inference_result={}

		# 입력받은 문장을 tokenizer를 사용하여 토큰화 및 수치화
		input_ids = tokenizer.batch_encode_plus([sentence], padding="longest", max_length=self.max_seq_length,
										pad_to_max_length=True, truncation=True, return_tensors='pt')

		# logger.info(input_ids)
		output_sequences = model.generate(input_ids=input_ids['input_ids'].to(self.device),
									attention_mask=input_ids['attention_mask'].to(self.device), num_beams=1, max_length=self.max_seq_length,early_stopping=True)
		# logger.info(output_sequences)
        
		output_label_str = [tokenizer.decode(sequence, skip_special_tokens=True, clean_up_tokenization_spaces=True) for sequence in output_sequences]
        # total_preds.extend(output_label_str)
        
		# logger.info(output_label_str)
  
		# 추론 서비스 토큰 개수
		input_token_count=getDataSetTokenCount(input_ids['input_ids'].tolist()[0])
		# logger.info(input_token_count)
		total_token_count=getDataSetTokenCount(input_ids['input_ids'].tolist()[0])
		# logger.info(total_token_count)
  
		# dict에 토큰 추가 0216
		inference_result["input_token_count"] = input_token_count
		inference_result["total_token_count"] = total_token_count

		inference_result_list.append({'result':output_label_str})
	
		inference_result["result"] = inference_result_list
  
		# dict에 result로 결과값 리스트를 저장 0216
		# inference_result["result"] = inference_result_list

  
		# del input_ids
		# del outputs
		# torch.cuda.empty_cache()
		# print(inference_result)
		return inference_result

	# pipeline 실행
	def __call__(self):

		# pipeline
		# 1. load_model
		model, tokenizer = self.load_model()

		# 2. inference
		sentence = ''
		while(sentence != '종료'):
				sentence = input('문장을 입력하세요 : ')
				inference_result = self.inference(model, tokenizer, sentence)
				print(inference_result)
