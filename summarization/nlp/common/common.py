from genericpath import exists
import os
import torch
import pandas as pd
import time
import json
import tarfile
import ast
import decimal
import logging
import signal
from nlp.common.nlpargs import TrainArguments
from rouge import Rouge

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

def createDirectory(directory): 
	if not os.path.exists(directory): 
		os.makedirs(directory)

#gpu 설정
def get_available_memory():
	gpu_count = torch.cuda.device_count()	
	gpu_mem_info={}
	gpu_device_info={}
	for gpu_id in range(gpu_count):
		gpu_mem_info[gpu_id]={'total':0,'available':0,'used':0}
		gpu_device_info[gpu_id]=torch.cuda.get_device_properties(gpu_id)
	
	for gpu_id in range(gpu_count):
		torch.cuda.set_device(gpu_id)
		(available,max)=torch.cuda.mem_get_info()
		used=max-available
		gpu_mem_info[gpu_id]['total']=max/1024/1024
		gpu_mem_info[gpu_id]['available']=available/1024/1024
		gpu_mem_info[gpu_id]['used']=used/1024/1024
	return (gpu_count,gpu_mem_info)

def get_max_available_mem_device():
	gpu_cnt,mem_info=get_available_memory()
	return_gpu_id=0
	return_mem_available=0
	for gpu_id in range(gpu_cnt):
		if mem_info[gpu_id]['available']>return_mem_available:
			return_gpu_id=gpu_id
			return_mem_available=mem_info[gpu_id]['available']
	return (return_gpu_id,return_mem_available)


def set_service_gpu():
	# gpu			
	gpus=[]
	target_gpu,available_memory=get_max_available_mem_device()
	gpus.append(target_gpu)

	global target_device
	
	if len(gpus)>0:
		target_device='cuda:{}'.format(target_gpu)
	else:
		target_device = 'cpu'

	return target_device

def get_config_parameter(train_config_json):
	# 변경 후 config일 경우
	if(train_config_json.get("params")):
		parameterList = train_config_json["params"]
		train_config_json = {}

		for i in range(len(parameterList)):
			train_config_json[parameterList[i]["param_name"]] = parameterList[i]["param_value"]
			
		return train_config_json
	# 변경 전 config일 경우
	else:
		return train_config_json

	# trainDataUrl, trainConfig, fineTunningPath
def set_config(train_data_path, trainConfig, save_finetunning_path):
	global train_config
	global train_config_json
	
	train_config_json = {}
	trainConfigArgs = TrainArguments()
	
	config_file_path = '/summarization/repository/trainConfig/config.json'
	
	# 기본 경로 정보 셋팅		
	# 파일명과 경로를 자름
	path = set_path_name_ext(save_finetunning_path)
	save_finetunning_path = set_path_name_ext(path)
 
	trainConfigArgs.downstream_model_dir = save_finetunning_path
	trainConfigArgs.pretrained_model_config_path = config_file_path
	trainConfigArgs.downstream_corpus_root_dir = train_data_path
 
	# if os.path.exists(config_file_path):
	# if trainConfig is not None:
	 
	# 	with open(config_file_path, 'w', encoding='utf-8') as file:
	# 		json.dump(ast.literal_eval(trainConfig), file, indent="\t")
   
	# 	train_config_json = get_config_parameter(ast.literal_eval(trainConfig))
  
	# 	# trainConfig 상세 값의 조건문 적용
	# 	train_config_json["early_stopping_flag"] = False if train_config_json['early_stopping_flag'] == 0 else True
	# 	learning_rate_config = train_config_json.get('learning_rate') if train_config_json['learning_rate'] else 5e-5

	# 	# trainConfigArgs.batch_size = config_type_compare(TrainArguments.batch_size, train_config_json["batch_size"])
	# 	trainConfigArgs.max_seq_length = config_type_compare(TrainArguments.max_seq_length, train_config_json["max_seq_length"])
	# 	trainConfigArgs.split_ratio = train_config_json["split_ratio"]
	# 	trainConfigArgs.epochs = config_type_compare(TrainArguments.epochs, train_config_json["epochs"])
	# 	trainConfigArgs.learning_rate = config_type_compare(TrainArguments.learning_rate, train_config_json["learning_rate"])
	# 	trainConfigArgs.early_stopping_flag = config_type_compare(TrainArguments.early_stopping_flag, train_config_json["early_stopping_flag"])
	# 	trainConfigArgs.early_stopping_patience = config_type_compare(TrainArguments.early_stopping_patience, train_config_json["early_stopping_patience"])
	# 	trainConfigArgs.gradient_accumulation_steps = config_type_compare(TrainArguments.gradient_accumulation_steps, train_config_json["gradient_accumulation_steps"])
        
	# 	trainConfigArgs.prefix_flag = TrainArguments.prefix_flag
	# 	trainConfigArgs.t2t_flag = TrainArguments.t2t_flag
	# 	trainConfigArgs.label_length = TrainArguments.label_length

	# else:
	# 	trainConfigArgs

	return trainConfigArgs

# trainConfig에서 자료형이 일치하지 않는 경우를 대비하여 자료형을 일치시킴
def config_type_compare(trainConfigType, trianConfigValue):
	try:
		resultValue = None
		# TrainArguments class의 default형과 trainConfig에서 자료형 일치 확인
		if type(trainConfigType) != type(trianConfigValue):
			if type(trainConfigType) == int:
				resultValue = int(trianConfigValue)
			elif type(trainConfigType) == float:
				resultValue = float(trianConfigValue)
			elif type(trainConfigType) == str:
				resultValue = str(trianConfigValue)
			elif type(trainConfigType) == bool:
				resultValue = bool(trianConfigValue)
		else:
			resultValue = trianConfigValue
		return resultValue
	except Exception as e:
		logger.error(e)
		
def train_state_json(startTime, percent, endTime, finetunning_path):
	trainState={}
	trainState['startTime'] = ''
	trainState['progress'] = 0
	trainState['status'] = 'created'
	trainState['endTime'] = ''
	
	# fine-tunning 경로는 하드코딩이 아닌 nexus로 변경할 것 
	try:
		train_state_path = os.path.join(finetunning_path, 'trainStatus', 'trainStatus.json')
		path = set_path_name_ext(train_state_path)
		createDirectory(path)
		#json 파일을 먼저 읽는다
		if os.path.exists(train_state_path):
			with open(train_state_path, 'r', encoding='utf-8') as file:
				trainState = json.load(file)

		#시작시간 none값 확인
		if startTime is not None:
			trainState['startTime'] =  time.strftime('%Y%m%d%H%M%S',time.localtime(startTime))
		
		#진행률 none값 확인
		if percent is not None:
			# 소주점 반올림 및 0 ~ 100 범위로 작성
			trainState['progress'] = round(percent*100)
			# 진행률이 1.0이면 학습이 종료이므로 trained상태 
			if trainState['progress'] == 100:
				trainState['status'] = 'trained'
			# 진행률이 1.0이 아니면 학습이 진행중이므로 training상태 
			elif trainState['progress'] == 0:
				trainState['status'] = 'created'
			else:
				trainState['status'] = 'training'

		#종료시간 none값 확인
		if endTime is not None:
			trainState['endTime'] =   time.strftime('%Y%m%d%H%M%S',time.localtime(endTime))

		with open(train_state_path, 'w', encoding='utf-8') as file:
			json.dump(trainState, file, indent="\t")
	except Exception as e:
		logger.error(e)

def epoch_time_json(epochDuration, finetunning_path):
	try:
		epochState={}
		epochState['epochDuration'] = 0
  
		# epoch 당 걸린시간 확인 위한 json 파일
		epoch_state_path = os.path.join(finetunning_path, 'trainStatus', 'epochDuration.json')
		epoch_path = set_path_name_ext(epoch_state_path)
		createDirectory(epoch_path)
  
		if os.path.exists(epoch_state_path):
			with open(epoch_state_path, 'r', encoding='utf-8') as file:
				epochState = json.load(file)
	
		# epoch 당 걸린시간 none값 확인
		if epochDuration is not None:
			epochState['epochDuration'] =  round(epochDuration)
   
		with open(epoch_state_path, 'w', encoding='utf-8') as file:
			json.dump(epochState, file, indent="\t")
	except Exception as e:
		logger.info(e)
		
	# 학습 리포트 필수 정보
def make_train_report(self, modelName, task, save_epoch, min_val_loss, trainer_state, epoch_state, start_time, end_time):
	try:
		train_report=trainer_state
  
		new_report={}
  
		new_report['startTime']=time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(start_time))
		new_report['endTime']=time.strftime('%Y.%m.%d %H:%M:%S',time.localtime(end_time))
		new_report['trainingDuration']=round(end_time-start_time, 1)
		
		new_report['minValLoss']=round(train_report['best_metric'], 4)
  
		#loss 제일 작은 에폭 구하기
		for i in range(len(trainer_state["log_history"])):
			if trainer_state["log_history"][i].get('eval_loss') != None:
				if trainer_state["log_history"][i]["eval_loss"]==trainer_state['best_metric'] :
					minValLossEpoch=trainer_state["log_history"][i]["epoch"]
					savedModelValF1=trainer_state["log_history"][i]["eval_f1_rouge"]
					savedModelValPrecision=trainer_state["log_history"][i]["eval_precision_rouge"]
					savedModelValRecall=trainer_state["log_history"][i]["eval_recall_rouge"]
	 
		new_report['minValLossEpoch']=int(round(minValLossEpoch, 4)) # best epoch

		new_report['savedModelValLoss']=round(train_report['best_metric'], 4)

		new_report['savedModelValAccuracy']= 0.00
  
		new_report['savedModelValF1']=round(savedModelValF1, 4)

		new_report['savedModelValPrecision']=round(savedModelValPrecision, 4)
		new_report['savedModelValRecall']=round(savedModelValRecall, 4)

		# 학습데이터의 전체 token개수를 저장
		new_report['totalToken']=self.totalToken
		new_report['trainToken']=self.trainToken
		new_report['validToken']=self.validToken
		
		# 학습데이터의 전체 Length 저장
		new_report['totalLength']=self.totalLength
		new_report['trainLength']=self.trainLength
		new_report['validLength']=self.validLength

		new_report['settings']={
			'maxValEpoch':self.train_config.epochs,
			'batchSize':self.train_config.batch_size,
			'earlyStoppingFlag':self.train_config.early_stopping_flag,
			'earlyStoppingPatience':self.train_config.early_stopping_patience,
			'modelName':modelName,
			'learningRate': format(decimal.Decimal(self.train_config.learning_rate), '.5f')
		}
		new_report['trainingDataInfo']={
			'fileSize':self.dataSetting.train_file_size,
			'numOfTrainData':self.dataSetting.train_size,
			'numOfValidationData':self.dataSetting.valid_size,
			'numOfTestData':self.dataSetting.test_size,
		}

		new_report['epochReport']=[]


		try:
			for idx in range(len(epoch_state['log_history'])):
				train_epoch = eval(epoch_state["log_history"][idx])
	
				if 'train_loss' in train_epoch :
					new_report['epochReport'].append({	
						'epoch': int(train_epoch['epoch']) -1 ,
						'trainAccuracy': 0, 
						'trainLoss': round(train_epoch['train_loss'], 4),
						'trainF1': round(train_epoch['train_f1_rouge'], 4),
					})		
			for idx2 in range(len(epoch_state['log_history'])):
				valid_epoch = eval(epoch_state['log_history'][idx2])
	
				if 'eval_loss' in valid_epoch : 
					for idx3 in range(len(new_report['epochReport'])):		            
						if int(valid_epoch['epoch']) -1 == new_report['epochReport'][idx3]['epoch'] :  
							new_report['epochReport'][idx3]['valLoss'] = round(valid_epoch['eval_loss'], 4)
							new_report['epochReport'][idx3]['valF1'] = round(valid_epoch['eval_f1_rouge'], 4)
							new_report['epochReport'][idx3]['valPrecision'] = round(valid_epoch['eval_precision_rouge'], 4)
							new_report['epochReport'][idx3]['valRecall'] = round(valid_epoch['eval_recall_rouge'], 4)
		
		except Exception as e:
			logger.info(e)
   
		new_report['gpuInfo']={
			'gpuName':self.gpu_info['name'],
			'gpuMem':round(self.gpu_info['total_memory'], 1)
		}
	
		# train_report json으로 저장
		report_path = ''
		report_path = os.path.join(self.train_config.downstream_model_dir, 'report', 'report.json')
		path = set_path_name_ext(report_path)
		createDirectory(path)

	
		with open(report_path, 'w', encoding='utf-8') as file:
			json.dump(new_report, file, indent="\t")

		return report_path

	except Exception as e:
		logger.info(e)

# 결과값이 int일 경우 int 값 리턴
def check_val_type(val):
	if type(val) == int :
		return float(val)
	elif type(val) == float:
		return val
	else :
		return val.tolist()

# 디렉토리 생성을 위한 경로 설정
def set_path_name_ext(org_path):
	path, files = os.path.split(org_path) #- 경로와 파일명을 분리
	return path

# T5 tar.gz 압축 
def T5_tar_filetunning_model(self, downstream_model_dir):

	# path = set_path_name_ext(finetunning_path)
	save_finetunning_tar_path = downstream_model_dir

	createDirectory(os.path.join(self.train_config.downstream_model_dir,'model'))

	save_finetunning_tar_path_full = os.path.join(self.train_config.downstream_model_dir,'model','fineModel.tar.gz')
	#압축을 한다
	try:
		with tarfile.open(save_finetunning_tar_path_full, 'w:gz') as tar:
			for file in os.scandir(save_finetunning_tar_path):
				tar.add(file, arcname=os.path.basename(file))

		#압축 후 파일 삭제 | lhy | 1123
		if os.path.exists(save_finetunning_tar_path):
			for file in os.scandir(save_finetunning_tar_path):
				os.remove(file.path)
	except Exception as e:
		logger.error(e)

# tar.gz 해제
def extractall_filetunning_model(finetunning_path):
	path = set_path_name_ext(finetunning_path)
	#압축을 푼다
	try:
		# 압축해제 파일
		tar = tarfile.open(finetunning_path)
		# 현재위치에 압축 해제
		tar.extractall(path=path)  
		tar.close()
	except Exception as e:
		logger.error(e)

	return path

# 메모리 누수 문제를 임시적으로 해결하는 방편 pid를 확인하여 kill
def pidKill():
	os.kill(os.getpid(), signal.SIGKILL)


# token개수
def getDataSetTokenCount(input):
	
	tokenLen = 0	

	# 0값을 제거
	remove_set = {0}
	# 반복문을 통한 0값 제거
	result = [i for i in input if i not in remove_set]

	tokenLen = 0	
	# 추가된 앞뒤 토큰을 제거
	tokenLen = len(result)-2
	return tokenLen

def createClassTestDataJson(downstream_model_dir, x_test, y_test):
	testJson = {}
	utteranceList = []
	for i in range(len(y_test)):
		utteranceJson = {}
		utteranceJson["utterance"] = x_test[i]
		utteranceJson["correctIntent"] = y_test[i]
		utteranceList.append(utteranceJson)
	
	testJson["utterances"] = utteranceList
	test_file_path = os.path.join(downstream_model_dir, "testData")
	createDirectory(test_file_path)
	test_file_path = os.path.join(test_file_path,"testData.json")
	
	with open(test_file_path, 'w', encoding='utf-8') as file:
		json.dump(testJson, file, indent="\t", ensure_ascii=False)

def get_rouge(total_preds, total_labels):
	rouge_scorer = Rouge()
	total_preds = [' '.join(pred) for pred in total_preds]
	total_labels = [' '.join(label) for label in total_labels]

	scores = rouge_scorer.get_scores(total_preds, total_labels, avg=True)

	scores = {score_type:scores['f'] for score_type, scores in scores.items()}

	return scores['rouge-l']
