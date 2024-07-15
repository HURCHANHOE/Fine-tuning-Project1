from nlp.common.nlpargs import TrainArguments
from sklearn.model_selection import train_test_split
from nlp.common.common import createClassTestDataJson
import random
import csv
import os
import pathlib

#학습 데이터가 분리되어 있지 않을 때, 데이터를 분리해서 Train용 MultiClassCorpus, Validation용 MultiClassCorpus를 제공
# class 명 변경
class DataSetting:
		def __init__(self, args, downstream_corpus_root_dir):
				corpus_path=os.path.join(downstream_corpus_root_dir)
				# train data path로 파일 확장자 확인
				path = pathlib.Path(corpus_path)

				with open(corpus_path,'r', encoding = 'UTF8') as f:
						lines_raw=f.readlines()
				lines=[]
				# path.suffix 확장자로 구분자 설정
				if path.suffix =='.csv':
						f = open(corpus_path, 'r', encoding='utf-8')
						lines_raw = csv.reader(f)
						for line in lines_raw:
								lines.append(line)
				else:
						for line in lines_raw:
								split_list=line.split('\t')
								lines.append([split_list[0],split_list[1].replace('\n','')])

				# self.examples = []
				# self.intentTags = []
				train_texts=[]
				labels=[]
				for (i, line) in enumerate(lines):
						text_a=line[args.text_idx]
						label=line[args.label_idx]
						train_texts.append(text_a)
						labels.append(label)
				# #임시 추가
				# train_texts = train_texts[:400]
				# labels = labels[:400]
		
						# self.intentTags.append(label)
				# self.intentTags = list(set(self.intentTags))

				#T5에서 사용하는 labelMap | lhy | 1121
				# self.label_map = {}

				# for i in range(len(self.intentTags)):
						# self.label_map[self.intentTags[i]] = i
   
				x_batchTest = []
				y_batchTest = []
  
				# 04/18 train valid test 3가지로 분리, split 수정
				if(type(args.split_ratio) == list):
						if(args.split_ratio[2] == 0):
								x_train,x_valid,y_train,y_valid = train_test_split(train_texts,labels, test_size=args.split_ratio[1], random_state=args.seed) # stratify 제거

						else:
								testRatio = args.split_ratio[2]
								validRatio =  args.split_ratio[1]

								# 03/22 train valid : test 분리
								x_trainValid,x_batchTest,y_trainValid,y_batchTest = train_test_split(train_texts,labels, test_size=testRatio,random_state=args.seed) # stratify 제거

								# testData.json 생성
								if(len(y_batchTest) > 0):
										createClassTestDataJson(args.downstream_model_dir, x_batchTest, y_batchTest)

								# 03/22 train : valid 분리
								x_train,x_valid,y_train,y_valid = train_test_split(x_trainValid,y_trainValid, test_size=validRatio,random_state=args.seed) # stratify 제거

				# train valid 2가지로 분리
				else:
						x_train,x_valid,y_train,y_valid=train_test_split(train_texts,labels,test_size=args.split_ratio,random_state=args.seed) # stratify 제거

				# x_train,x_valid,y_train,y_valid = train_test_split(train_texts,labels, test_size=args.split_ratio, random_state=args.seed, stratify=labels) # stratify 제거

				self.return_params={}
				self.return_params['train']=(x_train,y_train)
				self.return_params['valid']=(x_valid,y_valid)
		
				#학습 데이터 정보를 기록
				self.train_size=len(x_train)
				self.valid_size=len(x_valid)
				self.test_size=len(x_batchTest)
				self.train_file_size=os.path.getsize(corpus_path)
				print('Train Size:',self.train_size,'Valid Size:',self.valid_size,'Test Size:',self.test_size,'Train File Size:',self.train_file_size)

		# Lhy 추가
		def split_train_valid(self):
				# Lhy return 값 생성

				return self.return_params['train'], self.return_params['valid']

		# def get_examples(self, data_root_path, mode):
		#       if mode=='train':
		#               return self.trainData
		#       elif mode=='test':
		#               return self.validationData
		#       else:
		#               return self.trainData

#       def get_labelsMap(self):
#               return self.label_map

#       def get_labels(self):
#               return self.intentTags

#       @property
#       def num_labels(self):
#               return len(self.get_labels())