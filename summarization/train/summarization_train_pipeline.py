import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(os.path.dirname((os.path.abspath(os.path.dirname((__file__))))))
import pickle
import shutil
import glob
import json
import torch
import logging
import math, json
import time
from transformers import T5Tokenizer,AutoConfig,set_seed, T5TokenizerFast,T5ForConditionalGeneration
from transformers.adapters import LoRAConfig, ConfigUnion

from tokenizers import AddedToken
import numpy as np

from nlp.summarizationnlp.summarizationdataset import T5ClassificationDataSet_t2t_summarization

from nlp.summarizationnlp.summarizationtask import T5EncoderClassificationTask

from nlp.common.nlpargs import TrainArguments
from nlp.common.common import set_config
from nlp.common.common import get_max_available_mem_device
from nlp.common.common import make_train_report
from nlp.common.common import set_path_name_ext
from nlp.common.common import createDirectory
from nlp.common.common import T5_tar_filetunning_model
from nlp.common.common import train_state_json

from nlp.summarizationnlp.summarizationcorpus import DataSetting
from interface.trainInterface import trainPipelineInterface

import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

class SummarizationTrainPipeline(trainPipelineInterface):
	def __init__(self, train_data_path, config_file_path, save_finetunning_path, base_path):
		logger.info("train_pipeline start")
		# 즉, backward 패스 과정에서 오류가 발생 했을 때, 구체적으로 어떤 파일의 어떤 연산에서 발생했는지 그 traceback을 출력해준다.
		# torch.autograd.set_detect_anomaly(True)
		# train config파일을 세팅
		logger.info("train_config start")
		self.train_config = set_config(train_data_path, config_file_path, save_finetunning_path)
		
		createDirectory(self.train_config.downstream_model_dir)

		logger.info("train datasetting start")
		self.dataSetting = DataSetting(self.train_config, train_data_path)
		if self.train_config.random_flag == False:
						set_seed(self.train_config.seed)

		if self.train_config.do_lower_case:
						if self.train_config.do_lower_case == 'False':
										self.train_config.do_lower_case=False
		# gpu 세팅
		self.gpus=[]
		target_gpu,available_memory=get_max_available_mem_device()
		self.gpus.append(target_gpu)
		self.train_config.set_gpu_ids(self.gpus)
		logger.info("self.train_config.gpus")
		logger.info(self.train_config.gpus)
		self.gpu_info = {}
		self.gpu_info['name'] = target_gpu
		self.gpu_info['total_memory'] = available_memory
		logger.info("self.gpu_info")
		logger.info(self.gpu_info)

		#base path 설정 
		self.s_model = base_path

		self.label_length = self.train_config.label_length
		# 임시 추가 변경해야함 qlora로
		self.pelt_task_name="test"

	def prepare_data(self):
		logger.info('prepare data start')
		#1. tokenizer setting
		tokenizer = T5TokenizerFast.from_pretrained(self.s_model)
		logger.info('T5TokenizerFast end')

		#2. corpus를 train과 valid로 분리(분리비율은 config 파일에서 설정)
		trainData, validationData = self.dataSetting.split_train_valid()
		logger.info('split_train_valid end')

		#2-1. T5에 맞는 데이터 
		(x_train, y_train) = trainData
		(x_val, y_val) = validationData
        
		train_dataset = T5ClassificationDataSet_t2t_summarization(tokenizer, self.prefix_token, x_train, y_train, self.train_config.max_seq_length, self.label_length)
		valid_dataset = T5ClassificationDataSet_t2t_summarization(tokenizer, self.prefix_token, x_val, y_val, self.train_config.max_seq_length, self.label_length, self.prefix_flag)
		logger.info('T5ClassificationDataSet_t2t_summarization end')

		# totalTokenCount를 계산| lhy | 1219
		self.trainToken = train_dataset.getTokenCount()
		self.validToken = valid_dataset.getTokenCount()
		self.totalToken = self.trainToken + self.validToken

		# totalDataLength를 계산| lhy | 1219
		self.trainLength = train_dataset.getDataLength()
		self.validLength = valid_dataset.getDataLength()
		self.totalLength = self.trainLength + self.validLength

		logger.info('prepare data end')
		return tokenizer, train_dataset, valid_dataset

	def prepare_model(self, tokenizer, train_dataset, valid_dataset):

		logger.info('prepare model start')
		# 1. pretrained_model setting
		# 1-1. pretrained_config file
		config = AutoConfig.from_pretrained(self.s_model)
		logger.info('AutoConfig.from_pretrained end')

		# 1-2. config에 label 개수 
		config.dropout_rate=0.1
		config.max_length = self.label_length 

		# Set up adapter config 여기서 전체 어댑터 적용해볼까?
		lora_config = LoRAConfig(r=4, alpha=4, attn_matrices=["q", "k", "v"])
		peft_config = ConfigUnion(lora_config)

		# 1-3. config가 적용된 model 로딩

		logger.info("model start")
		self.model = T5ForConditionalGeneration.from_pretrained(self.s_model, config=config)
		logger.info("model end")

		#adapter model setting
		logger.info("model adapter start")
		self.model.add_adapter(self.pelt_task_name, config=peft_config)
		self.model.train_adapter(self.pelt_task_name)

		self.model.set_active_adapters(self.pelt_task_name)
		self.model.shared.weight.requires_grad, self.model.lm_head.weight.requires_grad = False, False
		logger.info("model adapter end")

		print(self.model.adapter_summary())
		print("peft_params: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
		logger.info("model end")

		# 2. trainer setting
		task = T5EncoderClassificationTask(self.model, self.train_config, tokenizer, train_dataset, valid_dataset)
		logger.info('T5EncoderClassificationTask end')

		trainer = task.get_trainer()
		logger.info('get_trainer end')
		# train Mode로 변환
		self.model.train()

		logger.info('prepare model end')
		return trainer, task, config, peft_config

	# 0210 수정
	def do_train(self, trainer):
		logger.info('train start')
		start_time = time.time()
		trainer.train()
		end_time = time.time()
		logger.info('train end')

		return start_time, end_time

	# 0210 원본
	def save_model(self, task, config, tokenizer, peft_config,start_time,end_time):
		logger.info('save model start')
		result = {}
		resultCode = 0
		# T5에 맞는 저장경로 설정 | lhy | 1123

		downstream_model_dir = os.path.join(self.train_config.downstream_model_dir, 'temp', 'peftPath') 


		# 1. learn_info 저장 정보
		try:
						learn_info_path = os.path.join(downstream_model_dir, 'learn_info.pkl')
						path = set_path_name_ext(learn_info_path)
						createDirectory(path)
						resultCode = 0
		except Exception as e:
						resultCode = 1
						logger.info(e)
		learn_info = {}
		modelName = "t5-small"
		learn_info['pretrained_model_name']=modelName
		learn_info['max_seq_length'] = self.train_config.max_seq_length
		learn_info['softmax_flag']=self.train_config.softmax

		# 1-2. learn info를 pickle로 저장
		with open(learn_info_path, 'wb') as fp:
						pickle.dump(learn_info, fp)
						logger.info('learn_info end')
		resultCode = 0
		task.model.config.save_pretrained(downstream_model_dir)
		tokenizer.save_pretrained(downstream_model_dir)

		tokeBytes_path = os.path.join(self.byte_path, 'bytes.t')
		save_tokeBytes_path = os.path.join(downstream_model_dir, 'bytes.t')
		shutil.copyfile(tokeBytes_path, save_tokeBytes_path)
		logger.info("save bytes.t")
		logger.info('save tokenizer end')


		# Config save
		with open(downstream_model_dir + '/peft_config.json', 'w') as f:
						json.dump(peft_config.to_dict(), f, ensure_ascii=False)
		logger.info("save peft_config")

		# Model save
		try:
				raw_state_dict = task.model.state_dict()
				save_train_state_dict = {}
				for key, value in raw_state_dict.items():
						if self.pelt_task_name in key or 'shared' in key or 'lm_head' in key or 'embed_tokens' in key:
								save_train_state_dict[key] = value
				torch.save(save_train_state_dict, downstream_model_dir + '/peft_model.pt')
				logger.info("save model")
		except Exception as e:
						logger.info(e)

		# task.model.save(downstream_model_dir) #peft_model.pt
		#save_epoch, min_val_loss 구하기
		try:
						#Load trainer_state.json
						path=os.path.join(self.train_config.downstream_model_dir, 'temp')
						ckpt_list = os.listdir(path)
						ckpt_list = sorted([file for file in ckpt_list if file.startswith("checkpoint")])
						trainer_state_path = os.path.join(path+'/'+ckpt_list[0])
						with open(trainer_state_path+'/trainer_state.json','r') as f:
										trainer_state=json.load(f)
						save_epoch=self.train_config.epochs
						min_val_loss = round(trainer_state['best_metric'], 4)
						logger.info("set trainer state")
		except Exception as e:
						logger.info(e)

		# 5. 학습결과 정보
		# make_train_report : API를 제공하여 학습결과를 json으로 저장
		try:
						epoch_trainer_state_path = os.path.join(self.train_config.downstream_model_dir, 'temp','test_report', 'report.json')
						with open(epoch_trainer_state_path, 'r') as f:
								epoch_state=json.load(f)
						train_report_path = ''
						train_report_path = make_train_report(self, modelName, task, save_epoch, min_val_loss, trainer_state, epoch_state, start_time, end_time)
						logger.info('make_train_report end')
						resultCode = 0
		except Exception as e:
						resultCode = 1
						logger.info(e)

		# finetunning model.tar 생성
		try: 
						T5_tar_filetunning_model(self, downstream_model_dir)
						logger.info('T5_tar_filetunning_model end')
						resultCode = 0
		except Exception as e:
						resultCode = 1
						logger.info(e)

		#checkpoint 파일 삭제
		try:
						if os.path.exists(trainer_state_path):
										for file in os.scandir(trainer_state_path):
														os.remove(file.path)
		except Exception as e:
						logger.info(e)
		logger.info('save model end')

		torch.cuda.empty_cache()
		logger.info('empty_cache end')
		import gc
		gc.collect()
		logger.info('gc.collect memory end')

		progress = 1.0
		train_state_json(None, progress, None, self.train_config.downstream_model_dir)

		if resultCode == 0:
						result['resultCode'] = resultCode
						result['resultReport'] = train_report_path
						return result
		else:
						result['resultCode'] = resultCode
						return result


	# pipeline 실행
	def __call__(self):
		result = {}
		# pipeline
		# 1. prepare dataset
		tokenizer, fast_tokenizer, train_dataset, valid_dataset, train_dataset_light = self.prepare_data()

		# 2. prepare model
		trainer, task, config, peft_config = self.prepare_model(tokenizer, fast_tokenizer, train_dataset, valid_dataset, train_dataset_light)

		# 3. run train
		start_time, end_time = self.do_train(trainer)

		# 4. save model, save result

		result = self.save_model(task, config, tokenizer, peft_config, start_time, end_time)

		return result
	
train_data_path='C:\Users\chanhoe.Hur\Finetuning_project1\summarization\data\개인및관계_1000.csv'
config_file_path='{"params": [{"param_name": "gradient_accumulation_steps","param_value": 8,"param_describe": "Number of updates steps to accumulate the gradients for, before performing a backward/update pass. (int)"},{"param_name": "max_seq_length","param_value": 512,"param_describe": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded. (int)​"},{"param_name": "split_ratio","param_value": [0.90,0.10,0.20],"param_describe": "Train-Validation Split ratio"},{"param_name": "epochs","param_value":1,"param_describe": "Total number of training epochs to perform. (int)​"},{"param_name": "learning_rate","param_value": 5.00E-04,"param_describe": "The initial learning rate for the optimizer. (float)​"},{"param_name": "early_stopping_flag","param_value": 1,"param_describe": "early stopping apply flag(0 = False, 1 = True)​"},{"param_name": "early_stopping_patience","param_value": 3,"param_describe": "Use with metric_for_best_model to stop training when the specified metric worsens for early_stopping_patience evaluation calls.(int, scope : 3~5)"}]}'
save_finetunning_path='/result' 
base_path='paust/pko-t5-small' 

FM = SummarizationTrainPipeline(train_data_path, config_file_path, save_finetunning_path, base_path)

trainResult = FM()

