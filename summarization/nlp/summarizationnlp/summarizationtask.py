from nlp.common.common import train_state_json
from nlp.common.common import epoch_time_json
from nlp.common.nlpmetrics import compute_metrics
from nlp.common.common import get_max_available_mem_device
from nlp.common.common import set_path_name_ext
from nlp.common.common import createDirectory
from nlp.common.nlpargs import TrainArguments
from nlp.common.common import set_service_gpu
from pytorch_lightning import LightningModule

from torch.optim.lr_scheduler import ExponentialLR
from transformers.optimization import AdamW

from transformers import Trainer, TrainingArguments, EarlyStoppingCallback, TrainerCallback, Seq2SeqTrainingArguments, Seq2SeqTrainer

import torch
import os,time,sys
from pathlib import Path
import json

import logging
from copy import deepcopy
import random 

import os, sys
path_to_transformers = "/usr/local/lib/python3.8/site-packages"
sys.path.append(path_to_transformers)

from transformers.utils import (
	cached_property,
	requires_backends,
)
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

#MetricRecorder class
class MetricRecorder(EarlyStoppingCallback):
	def __init__(self, args:TrainArguments):
		self.batch_cnt=0
		self.train_epoch_cnt=0
		self.validation_epoch_cnt=0

		self.min_val_loss=9999
		self.max_val_acc=0
		self.min_val_loss_epoch=0

		self.epoch_cnt=0

		self.args=args
		self.five_percent=0
		self.start_time=None
		self.allocated_memory_peak=0
		# self.val_confmat=[]
		self.processed_batch=0
		self.total_steps = 0

	def set_train_report_dict(self,train_report):
		self.metric_report=train_report

	def on_train_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') :
		#train 시작 시간 기록
		self.train_start_time=time.time()

		self.metric_report['startTime']=self.train_start_time
		self.metric_report['epochReport']=[]
		#train Total Step 기록
		# save model을 위해 10% 증가
		self.total_steps=round(self.args.epochs*(TrainingArguments.per_device_train_batch_size+TrainingArguments.per_device_eval_batch_size[0])*1.1)
		logger.info("self.total_steps")
		logger.info(self.total_steps)

		# 학습 상태 기록
		train_state_json(self.train_start_time, None, None, self.args.downstream_model_dir)
		#epoch report에서 Tensor항목들을 일반 숫자로 변환

	def on_train_end(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule') :
		self.train_end_time=time.time()
		self.metric_report['endTime']=self.train_end_time
		self.metric_report['trainingDuration']=self.train_end_time-self.train_start_time
		# 학습 상태 기록
		train_state_json(None, None, self.train_end_time, self.args.downstream_model_dir)

	def on_train_epoch_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
		self.epoch_start_time=time.time()

	def on_train_epoch_end(self, trainer, pl_module):
		acc=trainer.callback_metrics['acc'].item()
		val_acc=0 if not trainer.callback_metrics.get('val_acc') else trainer.callback_metrics['val_acc']
		loss=trainer.callback_metrics['loss'].item()
		val_loss=1 if not trainer.callback_metrics.get('val_loss') else trainer.callback_metrics['val_loss']
		f1=trainer.callback_metrics['f1'].item()
		val_f1=0 if not trainer.callback_metrics.get('val_f1') else trainer.callback_metrics['val_f1']

		device_stat=trainer.accelerator.get_device_stats([0])
		self.allocated_memory_peak=device_stat['allocated_bytes.all.peak'] if device_stat['allocated_bytes.all.peak']> self.allocated_memory_peak else self.allocated_memory_peak

		self.train_epoch_cnt+=1

	def on_validation_start(self, trainer: 'pl.Trainer', pl_module: 'pl.LightningModule'):
		pass

	def on_validation_epoch_end(self, trainer, pl_module):
		device_stat=trainer.accelerator.get_device_stats([0])
		self.allocated_memory_peak=device_stat['allocated_bytes.all.peak'] if device_stat['allocated_bytes.all.peak']> self.allocated_memory_peak else self.allocated_memory_peak
		self.epoch_end_time=time.time()
		self.metric_report['gpuPeakMem']=self.allocated_memory_peak

		acc=trainer.callback_metrics['acc'].item()
		val_acc= 0 if not trainer.callback_metrics.get('val_acc') else trainer.callback_metrics['val_acc']
		loss=trainer.callback_metrics['loss'].item()
		val_loss= 1 if not trainer.callback_metrics.get('val_loss') else trainer.callback_metrics['val_loss']
		f1=trainer.callback_metrics['f1'].item()
		val_f1=0 if not trainer.callback_metrics.get('val_f1') else trainer.callback_metrics['val_f1']

		val_precision = trainer.callback_metrics['val_precision']

		val_recall = trainer.callback_metrics['val_recall']

		val_epoch_end_time=time.time()
		elapsed_time=val_epoch_end_time-self.epoch_start_time
		self.metric_report['epochReport'].append({'epoch':self.epoch_cnt,'trainAccuracy': acc,'trainLoss': loss,'trainF1': f1, 'valAccuracy':val_acc,'valLoss':val_loss,'valF1':val_f1, 'valPrecision': val_precision, 'valRecall': val_recall, 'elapsedTime':elapsed_time})

		# min_val_loss를 epoch 0일때 값으로 처리
		if self.epoch_cnt == 0:
			self.min_val_loss = val_loss
			self.min_val_epoch=self.epoch_cnt
			self.metric_report['minValLoss']=self.min_val_loss
			self.metric_report['minValLossEpoch']=self.min_val_epoch
			self.metric_report['callback_metrics'] = trainer.callback_metrics

		# 최소 val_loss값을 적용
		else:
			if self.min_val_loss>val_loss:
				self.min_val_loss=val_loss
				self.min_val_epoch=self.epoch_cnt
				self.metric_report['minValLoss']=self.min_val_loss
				self.metric_report['minValLossEpoch']=self.min_val_epoch
				self.metric_report['callback_metrics'] = trainer.callback_metrics

		self.validation_epoch_cnt += 1
		self.epoch_cnt+=1


	def on_train_batch_end(self, trainer,p1_module,outputs,batch,batch_idx,unused=0):
		if batch_idx==0:
			self.batch_cnt=0
		self.processed_batch+=1
		self.batch_cnt+=1
		percent=self.processed_batch/self.total_steps
		new_five_percent=percent//0.01
		if new_five_percent!=self.five_percent:
			#1%씩 상태를 업데이트 함
			update_msg={'source':'PROGRESS_UPD','train_id':self.args.train_id,'train_percent':percent}
				#self.progress_queue.put(update_msg)
			# 학습 상태 기록
			train_state_json(None, percent, None, self.args.downstream_model_dir)
			self.five_percent=new_five_percent

	def on_validation_batch_end(self, trainer,p1_module,outputs,batch,batch_idx,unused=0):
		self.processed_batch+=1
		self.batch_cnt+=1
		percent=self.processed_batch/self.total_steps
		new_five_percent=percent//0.01
		if new_five_percent!=self.five_percent:
			#1%씩 상태를 업데이트 함
			update_msg={'source':'PROGRESS_UPD','train_id':self.args.train_id,'train_percent':percent}
			#self.progress_queue.put(update_msg)
			# 학습 상태 기록
			train_state_json(None, percent, None, self.args.downstream_model_dir)


	def get_min_val_loss(self):
		return self.min_val_loss
	def get_min_val_loss_epoch(self):
		return self.min_val_loss_epoch
	def get_max_val_acc(self):
		return self.max_val_acc
	def get_total_epoch(self):
		return self.epoch_cnt

# train loss, f1, acc 기록
class CustomCallback(TrainerCallback):
	def __init__(self, args, trainer, valid_dataset, args_config) -> None:
		super().__init__()
		self._trainer = trainer
		self._args = args
		self.valid_dataset = valid_dataset
		self.args_config = args_config
		self.total_steps = 0
		self.total_steps = self._args.num_train_epochs*1.1
		self.processed_batch=0
		path1 = Path(self._args.output_dir)
		self.fine_path = path1.parent
		self.epochStateCnt =0

  
	# 교육 시작시 호출되는 이벤트
	def on_train_begin(self, args, state, control, **kwargs):
		logger.info("학습 시작")
		self.train_start_time=time.time()
		logger.info("학습 시작 시간")
		logger.info(self.train_start_time)
			
	def on_epoch_begin(self, args, state, control, **kwargs):
		self.epochStateCnt += 1
		try:
			if self.epochStateCnt == 2 :
				logger.info('epochDuration save start')
				# 학습 시작시간과 2에폭이 시작할때의 시간을 뺴서 총 에폭 하나가 돈 시간을 측정
				self.epoch_start_time=time.time()
				epochDuration = self.epoch_start_time - self.train_start_time
				logger.info('epochDuration')
				logger.info(epochDuration)

				epoch_time_json(epochDuration, self.args_config.downstream_model_dir)
			else:
				pass
		except Exception as e:
			logger.info(e)

		train_state_json(self.train_start_time, None, None, self.fine_path)
		logger.info("epoch start")

	def on_epoch_end(self, args, state, control, **kwargs):
		self.train_end_time=time.time()
		self.processed_batch+=1
		logger.info("epoch end")

		# if control.should_evaluate:
		# 	control_copy = deepcopy(control)
		# 	# train loss, train accuracy, train f1 ...
		# 	logger.info("trainer train dataset eval start")
		# 	#train data에 대한 evaluate start
		# 	train_info = self._trainer.evaluate(eval_dataset=self.valid_dataset, metric_key_prefix="valid")

		# 	# 필요한 정보 추출하여 사용 가능
		# 	train_loss, train_f1 = train_info['train_loss'], train_info['train_f1_rouge']
		# 	logger.info("train_loss")
		# 	logger.info(train_loss)
		# 	logger.info("epoch")
		# 	logger.info(self.processed_batch)
		# 값 이상시 수정 필요
		percent= self.processed_batch/self.total_steps

		train_state_json(None, percent, self.train_end_time, self.fine_path)
    
	def on_evaluate(self, args, state, control,**kwargs):
		try: 
	  
			logger.info("epoch history 저장")
			testList = []
			testDict = {}
			testDict["log_history"] = []
   
			# state json 저장
			for i in range(len(state.log_history)):
				data = state.log_history[i]
				json_data = json.dumps(data)
				testDict["log_history"].append(json_data)

			report_path = os.path.join(self._args.output_dir, 'test_report', 'report.json')
			path = set_path_name_ext(report_path)
			createDirectory(path)
			with open(report_path, 'w', encoding='utf-8') as outfile:
				json.dump(testDict, outfile, indent="\t")
	  
		except Exception as e:
			logger.info(e)

class customTrainingArguments(Seq2SeqTrainingArguments):
	def __init__(self,*args, **kwargs):
		super(customTrainingArguments, self).__init__(*args, **kwargs)

	@cached_property
	def _setup_devices(self) -> "torch.device":
		logger.info("localrank 확인")
		logger.info(self.local_rank)
		
		requires_backends(self, ["torch"])
		logger.info("PyTorch: setting up devices")
		if torch.distributed.is_available() and torch.distributed.is_initialized() and self.local_rank == -1:
			logger.warning(
				"torch.distributed process group is initialized, but local_rank == -1. "
				"In order to use Torch DDP, launch your script with `python -m torch.distributed.launch"
			)
		
		elif self.local_rank == -1:
			logger.info("local_rank==-1 사용")
			if self.use_mps_device:
				logger.info("mps 사용")
				if not torch.backends.mps.is_available():
					if not torch.backends.mps.is_built():
						raise AssertionError(
							"MPS not available because the current PyTorch install was not "
							"built with MPS enabled. Please install torch version >=1.12.0 on "
							"your Apple silicon Mac running macOS 12.3 or later with a native "
							"version (arm64) of Python"
						)
					else:
						raise AssertionError(
							"MPS not available because the current MacOS version is not 12.3+ "
							"and/or you do not have an MPS-enabled device on this machine."
						)
				else:
					logger.info("mps 사용 22")

			else:
				target_gpu,available_memory=get_max_available_mem_device()
				logger.info("gpu id 확인")
				logger.info(target_gpu)
				device = torch.device("cuda:{}".format(target_gpu) if torch.cuda.is_available() else "cpu")

				self._n_gpu = 1
		else:
			logger.info("local_rank==-1 사용 안함")
			if not torch.distributed.is_initialized():
				torch.distributed.init_process_group(backend="nccl", timeout=self.ddp_timeout_delta)
			device = torch.device("cuda", self.local_rank)
			self._n_gpu = 1

		if device.type == "cuda":
			logger.info("device cuda 사용")
			torch.cuda.set_device(device)

		return device

	@property
	def device(self) -> "torch.device":
		"""
		The device used by this process.
		"""
		requires_backends(self, ["torch"])
		return self._setup_devices

	@property
	def n_gpu(self):
		"""
		The number of GPUs used by this process.

		Note:
			This will only be greater than one when you have multiple GPUs available but are not using distributed
			training. For distributed training, it will always be 1.
		"""
		requires_backends(self, ["torch"])
		# Make sure `self._n_gpu` is properly setup.
		_ = self._setup_devices
		return self._n_gpu
	
#Task Class
class T5EncoderClassificationTask(LightningModule):
	def __init__(self,target_model, args: TrainArguments, tokenizer, fast_tokenizer, train_dataset, valid_dataset):
		super().__init__()
		self.model=target_model
		self.args = args
		self.tokenizer=tokenizer
		self.fast_tokenizer=fast_tokenizer
		self.train_dataset=train_dataset
		self.valid_dataset=valid_dataset
  
		test = compute_metrics(self.tokenizer, self.fast_tokenizer)
		if self.args.t2t_flag==True:
			self.compute_metrics = test.compute_metrics_rouge
		self.metric_report=None
		logger.info("self.args.early_stopping_flag")
		logger.info(self.args.early_stopping_flag)
        
		if self.args.early_stopping_flag == False :
			self.args.peft_monitor = "eval_loss"
			early_stopping_patience_false = self.args.epochs+1
			self.callback_list = [EarlyStoppingCallback(early_stopping_patience=early_stopping_patience_false)]
		else :
			self.args.peft_monitor = "loss"
			self.callback_list = [EarlyStoppingCallback(early_stopping_patience=self.args.early_stopping_patience)]
  
	def configure_optimizers(self):
		optimizer=torch.optim.AdamW(self.parameters(),lr=self.args.learning_rate)
		scheduler=ExponentialLR(optimizer,gamma=self.args.adam_epsilon)
		return [optimizer],[scheduler]

	def training_step(self, batch, batch_idx):
		outputs=self.model(**batch)
		loss=outputs.loss
		preds=outputs.logits.argmax(dim=-1)
		labels=batch["labels"]

		acc=multiclass_accuracy(preds,labels)

		#f1 Score
		f1Score = multiclass_f1Score(preds, labels, self.model.num_labels)

		self.log("loss", loss, prog_bar=False, logger=True, on_step=True, on_epoch=False)
		self.log("acc", acc, prog_bar=True, logger=True, on_step=True, on_epoch=False)
		self.log("f1", f1Score, prog_bar=True, logger=True, on_step=True, on_epoch=False)

		return loss

	def validation_step(self,batch,batch_idx):
		outputs=self.model(**batch)
		loss=outputs.loss
		preds=outputs.logits.argmax(dim=-1)
		labels = batch["labels"]

		acc = multiclass_accuracy(preds, labels)

		#f1 Score
		f1Score = multiclass_f1Score(preds, labels, self.model.num_labels)
		#precisionScore recallScore
		precisionScore, recallScore = multiclass_preciRecallScore(preds, labels, self.model.num_labels)


		#confusion Matrix
		# val_confmat =  multiclass_confMatrix(preds, labels, self.model.num_labels)
		# self.metric_report['val_confmat'] = val_confmat

		self.log("val_loss", loss, prog_bar=True, logger=True, on_step=False, on_epoch=True)
		self.log("val_acc", acc, prog_bar=True, logger=True, on_step=False, on_epoch=True)
		self.log("val_f1", f1Score, prog_bar=True, logger=True, on_step=False, on_epoch=True)
		self.log("val_precision", precisionScore, prog_bar=True, logger=True, on_step=False, on_epoch=True)
		self.log("val_recall", recallScore, prog_bar=True, logger=True, on_step=False, on_epoch=True)

		return loss

	def get_train_report(self):
		return self.metric_report

	def get_trainer(self, return_trainer_only=True):
		downstream_model_dir = os.path.join(self.args.downstream_model_dir, 'temp')
		ckpt_path = os.path.abspath(downstream_model_dir)
		os.makedirs(ckpt_path, exist_ok=True)
		args_monitor=self.args.peft_monitor
		args_mode = self.args.mode

		logger.info("self.args.gpus get Trainer")
		logger.info(self.args.gpus)
  
		TrainingArguments = customTrainingArguments(
		output_dir = ckpt_path,
		evaluation_strategy="epoch",
		save_strategy="epoch",
		learning_rate=self.args.learning_rate,
		per_device_train_batch_size=self.args.batch_size,
		gradient_accumulation_steps=self.args.gradient_accumulation_steps,
		# gradient_accumulation_steps = 8,
		eval_accumulation_steps=8,
		per_device_eval_batch_size=self.args.batch_size,
		num_train_epochs=self.args.epochs,
		weight_decay=0.01,
		warmup_ratio=0.1,
		logging_steps=100,
		load_best_model_at_end=True,
		metric_for_best_model=args_monitor,
		save_total_limit=1,
		# deepspeed=self.args.deepspeed,
		fp16=self.args.fp16,
		greater_is_better=False if args_mode == 'min' else True,
		predict_with_generate=True
		)

  
		# callback_list = [EarlyStoppingCallback(early_stopping_patience=self.args.early_stopping_patience)]
		trainer = Seq2SeqTrainer(
			self.model,
			TrainingArguments,
			train_dataset=self.train_dataset,
			eval_dataset=self.valid_dataset,
			tokenizer=self.tokenizer,
			compute_metrics=self.compute_metrics,
			callbacks=self.callback_list

		)
		trainer.add_callback(CustomCallback(TrainingArguments, trainer, self.valid_dataset, self.args))
  
		# self.metric_report=trainer.compute_metrics
		# metric_recorder.set_train_report_dict(self.metric_report)
		# trainer = Trainer(
		#                       max_epochs=self.args.epochs,
		#                       fast_dev_run=self.args.test_mode,
		#                       num_sanity_val_steps=None if self.args.test_mode else 0,
		#                       callbacks=callback_list,
		#                       default_root_dir=ckpt_path,
		#                       deterministic=True,
		#                       gpus=self.args.gpus,
		#                       precision=32,
		#                       #precision=16,
		#                       tpu_cores=None,
		# )
		if return_trainer_only:
			return trainer
		else:
			return trainer
