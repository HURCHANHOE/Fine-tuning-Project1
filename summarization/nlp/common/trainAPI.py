import os, sys
import logging
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
import json
import asyncio
from fastapi import FastAPI

from trainListener import AsyncioRabbitMQTrain
from testListener import AsyncioRabbitMQTest

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

app = FastAPI()
ep = None

@app.on_event("startup")
async def startup():
	global ep_train
	global ep_test
	await asyncio.sleep(10) # Wait for MQ

	# 학습 listener 생성
	ep_train = AsyncioRabbitMQTrain()
	try:
		logger.info('train connect run start')
		ep_train.connect()
		logger.info('train connect run end')
	except Exception as e:
		logger.info(e)	
	
	await asyncio.sleep(10) # Wait for MQ	
	# 성능 테스트 listener 생성
	ep_test = AsyncioRabbitMQTest()
	try:
		logger.info('test connect run start')
		ep_test.connect()
		logger.info('test connect run end')
	except Exception as e:
		logger.info(e)
		
@app.get("/009/getTrainConfig")
async def get_classification_train_config():
	
	train_config_path = '/summarization/repository/trainConfig/config.json'
	trainConfig = {}
	
	#json 파일을 먼저 읽는다
	if os.path.exists(train_config_path):
		with open(train_config_path, 'r', encoding='utf-8') as file:
			trainConfig = json.load(file)

	return trainConfig
