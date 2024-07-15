import os, sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname((__file__)))))))
import datetime
import json
import pika
import functools
import logging
import time
from pika.exchange_type import ExchangeType
from pika.adapters.asyncio_connection import AsyncioConnection
from pika.exchange_type import ExchangeType
from fastapi import FastAPI
from pydantic import BaseModel

from typing import Any, Dict

from minio import Minio
from ktnlp.common.common import extractall_filetunning_model
import shutil

import asyncio, json, os, queue, threading
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
# sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname((__file__)))))))

from service.summarization_service_pipeline import SummarizationServicePipeline

FM = SummarizationServicePipeline()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

instanceid = ''
instanceid = os.environ['instanceid']

queueName = 'ModelLoad_' + instanceid

routingKey = 'ModelLoadResult.all.load'

rabbitMqUrl = os.environ['rabbitMqUrl']
rabbitMqVhost = os.environ['rabbitMqVhost']
rabbitMqPort = os.environ['rabbitMqPort']
rabbitMqId = os.environ['rabbitMqId']
rabbitMqPw = os.environ['rabbitMqPw']

MinIOEndpoint = os.environ['endpoint']
MinIOAccessKey = os.environ['accesskey']
MinIOSecretKey = os.environ['secretkey']

try :
	client = Minio(MinIOEndpoint, access_key = MinIOAccessKey, secret_key = MinIOSecretKey, secure = False)
	logger.info(client.list_buckets())
except Exception as e :
	logger.info('minio error')
	logger.info(e)

logger.info('minio connect finish')

class Utterance(BaseModel):
	utterance: str
	
class AsyncioRabbitMQ(object):
	EXCHANGE = 'nlu-topic-exchange'
	# EXCHANGE_TYPE = ExchangeType.topic
	# PUBLISH_INTERVAL = 1
	QUEUE = queueName
	ROUTING_KEY = routingKey
	_message_number = 0
	_deliveries = []
	model = None
	tokenizer = None

	client = client
	logger.info('AsyncioRabbitMQ class start')

	def __init__(self):
		self._connection = None
		self._channel = None
		self._url = rabbitMqUrl
		self._vhost = rabbitMqVhost
		self._port = rabbitMqPort
		self._cred = pika.PlainCredentials(rabbitMqId, rabbitMqPw)
		self._acked = 0
		self._nacked = 0

		self._stopping = False
		# self._url = amqp_url
	def connect(self):
		logger.info('Connecting to %s', self._url)
		logger.info('AsyncioConnection start')
		return AsyncioConnection(
			pika.ConnectionParameters(self._url, 
									self._port, 
									self._vhost, 
									self._cred, 
									heartbeat=0,
									tcp_options={'TCP_KEEPIDLE':60}), # connection error timeout)

			on_open_callback=self.on_connection_open,
			on_open_error_callback=self.on_connection_open_error,
			on_close_callback=self.on_connection_closed)

	def on_connection_open(self, connection):
		logger.info('Connection opened')
		self._connection = connection
		logger.info('Creating a new channel')
		self._connection.channel(on_open_callback=self.on_channel_open)

	def on_connection_open_error(self, _unused_connection, err):
		logger.error('Connection open failed: %s', err)

	def on_connection_closed(self, _unused_connection, reason):
		logger.warning('Connection closed: %s', reason)
		self._channel = None

	def on_channel_open(self, channel):
		logger.info('Channel opened')
		self._channel = channel
		self.add_on_channel_close_callback()
		self.start_publishing()
		self.main()

	def add_on_channel_close_callback(self):
		logger.info('Adding channel close callback')
		self._channel.add_on_close_callback(self.on_channel_closed)

	def on_channel_closed(self, channel, reason):
		logger.warning('Channel %i was closed: %s', channel, reason)
		self._channel = None
		if not self._stopping:
			self._connection.close()

	def start_publishing(self):
		logger.info('Issuing Confirm.Select RPC command')
		self._channel.confirm_delivery(self.on_delivery_confirmation)

	def on_delivery_confirmation(self, method_frame):
		confirmation_type = method_frame.method.NAME.split('.')[1].lower()
		logger.info('Received %s for delivery tag: %i', confirmation_type, method_frame.method.delivery_tag)
		if confirmation_type == 'ack':
			self._acked += 1
		elif confirmation_type == 'nack':
			self._nacked += 1
		if len(AsyncioRabbitMQ._deliveries) != 0:
			AsyncioRabbitMQ._deliveries.remove(method_frame.method.delivery_tag)
		logger.info(
			'Published %i messages, %i have yet to be confirmed, '
			'%i were acked and %i were nacked', AsyncioRabbitMQ._message_number,
			len(AsyncioRabbitMQ._deliveries), self._acked, self._nacked)

	def main(self):
		logger.info('Consumer is starting...')
		self._channel.basic_consume(
			queue = self.QUEUE, 
			on_message_callback = AsyncioRabbitMQ.on_message,
			# 메세지 처리 완료 메세지 승인을 알리기 위해 소비자에게 다시 보냄 이는 메세지 손실을 방지하기 위해 사용함 | lhy | 1121
			auto_ack = True
		)
		logger.info('Consumer is end...')
		return 
		
	def on_message(channel, method_frame, header_frame, body):
		
		logger.info('on_message method start')
		try:
			# message 생성
			result = {}
			param = {}
			resultCode = 0
			# base_path= '/minio/BaseEngine/kt-midm-11b'
			logger.info('on_message body start')
			logger.info(body)
			message = str(body, "utf-8")
			if message != '':
				try:
					logger.info('on_message param start')
					# 학습 전에 body message(str)를 dict로 변환
					param = json.loads(message, strict=False)
					# ack 수동 ***(queue ack)***
					# channel.basic_ack(delivery_tag=method_frame.delivery_tag)
					logger.info('on_message param end')
				except Exception as e:
					result['resultCode'] = '7103'
					result['result'] = '파일 업로드 실패 : ' + str(e)
				try:
					logger.info('sevice run start')
					logger.info(param)
	 
					# # minio
					path = '/minio/BaseEngine'
					bucketName = param['baseModelNm']
					# version = param['baseModelVersion']

					# save_path = os.path.join(path, bucketName, version)
					save_path = os.path.join(path, bucketName)

					# createDirectory(save_path)
					if os.path.exists(save_path) :
							shutil.rmtree(save_path)
							os.makedirs(save_path)
					else:
							os.makedirs(save_path)

					client = AsyncioRabbitMQ.client
					files = client.list_objects(bucketName, use_url_encoding_type = 'utf-8', recursive = True)
					for _file in files :
						out_path = os.path.join(path, bucketName, _file.object_name)
						logger.info("out_path")
						logger.info(out_path)
						client.fget_object(bucketName, _file.object_name, out_path)
						filename = _file

						_list = os.listdir(save_path)
						logger.info("_list")
						logger.info(_list)
						if len(_list) == 1 :
								base_path = extractall_filetunning_model(out_path)
								logger.info('base_path')
								logger.info(base_path)

								for item in os.listdir(base_path) :
										sub_path = os.path.join(base_path, item)

										if os.path.isdir(sub_path) == True :
												baseengine_path = sub_path
												break

										else :
												baseengine_path = base_path

						else :
								baseengine_path = save_path

						logger.info('output create')
						logger.info(baseengine_path)
					# baseengine_path = '/nlu/BaseEngine/T5/kt-ulm-11b-no16'
					AsyncioRabbitMQ.model, AsyncioRabbitMQ.tokenizer = FM.load_model(param['fineTunningPath'], baseengine_path)
					logger.info('sevice run end')
					# shutil.rmtree(baseengine_path)
					
					AsyncioRabbitMQ._message_number += 1
					AsyncioRabbitMQ._deliveries.append(AsyncioRabbitMQ._message_number)
					logger.info('Published message # %i', AsyncioRabbitMQ._message_number)
					
					result['callbackUrl'] = param['callbackUrl']
					result['serviceInstanceId'] = param['serviceInstanceId']
					result['resultCode'] = '0000'
					result['result'] = '정상 응답'
			
				except Exception as e:
					result['callbackUrl'] = param['callbackUrl']
					result['serviceInstanceId'] = param['serviceInstanceId']
					result['resultCode'] = '7103'
					result['result'] = '파일 업로드 실패 : ' + str(e)

			else:			
				result['callbackUrl'] = param['callbackUrl']
				result['serviceInstanceId'] = param['serviceInstanceId']
				result['resultCode'] = '7103'
				result['result'] = '파일 업로드 실패 : ' + str(e)


		except Exception as e:
			result['resultCode'] = '7103'
			result['result'] = '파일 업로드 실패 : ' + str(e)
		
		finally:
			# instance_id로 routing key 설정
			# routeKey= 'TrainResult.'+instance_id+'.test'
						# dict -> str로 변환
			message = ''
			message = json.dumps(result)
			logger.info('message send start')

			logger.info(AsyncioRabbitMQ.EXCHANGE)
			logger.info(AsyncioRabbitMQ.ROUTING_KEY)
			logger.info(result)
			
			logger.info('basic_publish run start')
			channel.basic_publish(AsyncioRabbitMQ.EXCHANGE, AsyncioRabbitMQ.ROUTING_KEY, message)
			logger.info('basic_publish run end')
			return

app = FastAPI()
ep = None

@app.on_event("startup")
async def startup():
	global ep
	await asyncio.sleep(10) # Wait for MQ
	ep = AsyncioRabbitMQ()
	try:
		logger.info('connect run start')
		ep.connect()
		logger.info('connect run end')
	except Exception as e:
		logger.info(e)

# 입력받은 문자에 대한 의도 
@app.post("/009/serviceInference")
async def inference(utterance : Utterance):
	logger.info(utterance.utterance)
	inference_result = []
	try:
		inference_result = FM.inference(ep.model, ep.tokenizer, utterance.utterance)
		
	except Exception as e:
		logger.info(e)
	logger.info(inference_result)
	return inference_result

