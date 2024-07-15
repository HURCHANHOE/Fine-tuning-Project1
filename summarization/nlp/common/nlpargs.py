from dataclasses import dataclass,field
from glob import glob
import os
from statistics import mode

@dataclass
class TrainArguments:
	pretrained_model_name: str = field(default="beomi/kcbert-base",metadata={"help": "pretrained model name"})
	downstream_task_name: str = field(default="document-classification",metadata={"help": "The name of the downstream data."})
	downstream_corpus_name: str = field(default=None,metadata={"help": "The name of the downstream data."})
	downstream_corpus_root_dir: str = field(default="/root/Korpora",metadata={"help": "The root directory of the downstream data."})
	downstream_model_dir: str = field(default="None",metadata={"help": "The output model dir."})
	max_seq_length: int = field(default=512,metadata={"help": "The maximum total input sequence length after tokenization. Sequences longer than this will be truncated, sequences shorter will be padded."})
	save_top_k: int = field(default=1,metadata={"help": "save top k model checkpoints."})
	monitor: str = field(default="min val_loss",metadata={"help": "monitor condition (save top k)"})
	seed: int = field(default=7,metadata={"help": "random seed."})
	overwrite_cache: bool = field(default=False,metadata={"help": "Overwrite the cached training and evaluation sets"})
	force_download: bool = field(default=False,metadata={"help": "force to download downstream data and pretrained models."})
	test_mode: bool = field(default=False,metadata={"help": "Test Mode enables `fast_dev_run`"})
	learning_rate: float = field(default=1e-5,metadata={"help": "learning rate"})
	epochs: int = field(default=2,metadata={"help": "max epochs"})
	batch_size: int = field(default=1,metadata={"help": "batch size. if 0, Let PyTorch Lightening find the best batch size"})
	cpu_workers: int = field(default=os.cpu_count(),metadata={"help": "number of CPU workers"})
	# fp16: bool = field(default=False,metadata={"help": "Enable train on FP16"})
	tpu_cores: int = field(default=0,metadata={"help": "Enable TPU with 1 core or 8 cores"})
	split_ratio: float=field(default=0.1,metadata={"help":"Train-Validation Split ratio"})
	early_stopping_patience : int=field(default=2,metadata={"help":"Early Stopping Patiente"})
	text_idx:int=field(default=0,metadata={"help":"utterance text column"})
	label_idx: int = field(default=1, metadata={"help": "intent text column"})
	softmax:bool=field(default=False,metadata={"help":"Is Final layer softmax?"})
	pretrained_vocab_path:str=field(default=None,metadata={"help":"pretrained model vocab path"})
	pretrained_model_config_path:str=field(default=None,metadata={"help":"pretrained model config path"})
	pretrained_model_path:str=field(default=None,metadata={"help":"pretrained model path"})
	train_id:str=field(default=None,metadata={"help":"train_id"})

	# classification만 있는 내용
	early_stopping_flag:bool=field(default=True,metadata={"help":"early stopping apply flag"})
	sys_oob_flag:bool=field(default=False,metadata={"helo":"system oob set"})
	sys_oob_data_path:str=field(default=None,metadata={"help":"system oob data path"})
	lr_gamma:float=field(default=0.9,metadata={"help":"learning rate gamma"})
	base_volume:str=field(default="/data_volume/",metadata={"help":"Base Volume"})
	
	# T5 내용 추가 | lhy | 1121
	t5_monitor: str = field(default="val_loss",metadata={"help": "monitor condition (save top k)"})
	mode: str = field(default="min",metadata={"help": "mode"})
	adam_epsilon:float=field(default=0.99,metadata={"help":""})
	tokenizers_parallelism:bool=field(default=False,metadata={"help":""})
	label_length: int = field(default=128, metadata={"help": ""})
	prefix_flag:bool=field(default=False,metadata={"help":""})
	prefix_token: str = field(default="classification:",metadata={"help": ""})
	
	
	random_flag:bool=field(default=False,metadata={"help":""})
	distribute:bool=field(default=False,metadata={"help":""})
	do_lower_case:bool=field(default=True,metadata={"help":""})
 
	#peft 추가
	peft_monitor: str = field(default="loss",metadata={"help": "monitor condition (save top k)"})
	deepspeed:str = field(default="/nlu/ds_config/ds_config_zero2.json",metadata={"help": "deepseed path"})
	fp16:bool=field(default=True,metadata={"help":"fp16"})
	t2t_flag:bool=field(default=True,metadata={"help":""})
	pelt_task_name:str=field(default="kt_peft", metadata={"help":""})
    
	# mtft 파라미터 추가
	gradient_accumulation_steps: int = field(default=8, metadata={"help": ""})
	def set_gpu_ids(self,gpu_ids):
		self.gpus=gpu_ids

	def set_log_queue(self,log_queue):
		self.log_queue=log_queue
