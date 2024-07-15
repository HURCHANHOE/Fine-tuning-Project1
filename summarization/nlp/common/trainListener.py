import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname((__file__)))))))
import logging
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
import json
import shutil

from train.summarization_train_pipeline import SummarizationTrainPipeline
from common import pidKill
from nlp.common.common import extractall_filetunning_model
from nlp.common.common import createDirectory
from minio import Minio

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

instanceid = ''
instanceid = os.environ['instanceid']

trainQueueName = 'Train_' + instanceid
trainRoutingKey = 'TrainResult.all.train'

rabbitMqUrl = os.environ['rabbitMqUrl']
rabbitMqVhost = os.environ['rabbitMqVhost']
rabbitMqPort = os.environ['rabbitMqPort']
rabbitMqId = os.environ['rabbitMqId']
rabbitMqPw = os.environ['rabbitMqPw']

MinIOEndpoint = os.environ['endpoint']
MinIOAccessKey = os.environ['accesskey']
MinIOSecretKey = os.environ['secretkey']

logger.info(MinIOEndpoint)
logger.info(MinIOAccessKey)
logger.info(MinIOSecretKey)

try :
	client = Minio(MinIOEndpoint, access_key = MinIOAccessKey, secret_key = MinIOSecretKey, secure = False)
	logger.info(client.list_buckets())
except Exception as e :
	logger.info('minio error')
	logger.info(e)

logger.info('minio connect finish')

class AsyncioRabbitMQTrain(object):
	EXCHANGE = 'nlu-topic-exchange'
	# EXCHANGE_TYPE = ExchangeType.topic
	# PUBLISH_INTERVAL = 1
	QUEUE = trainQueueName
	ROUTING_KEY = trainRoutingKey
	autoTest = False
	
	_message_number = 0
	_deliveries = []

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
		if len(AsyncioRabbitMQTrain._deliveries) > 0:
			AsyncioRabbitMQTrain._deliveries.remove(method_frame.method.delivery_tag)
		logger.info(
			'Published %i messages, %i have yet to be confirmed, '
			'%i were acked and %i were nacked', AsyncioRabbitMQTrain._message_number,
			len(AsyncioRabbitMQTrain._deliveries), self._acked, self._nacked)
			
		# 메모리 누수 문제를 임시적으로 해결하는 방편 pid를 확인하여 kill | lhy | 1220
		# 자동분할이 있는 경우 pid kill하지 않음
		if AsyncioRabbitMQTrain.autoTest == False:
			pidKill()


	def main(self):
		logger.info('main is starting...self.QUEUE is : ')
		logger.info(self.QUEUE)

		self._channel.basic_consume(
			queue = self.QUEUE, 
			on_message_callback = AsyncioRabbitMQTrain.on_message,
			# 메세지 처리 완료 메세지 승인을 알리기 위해 소비자에게 다시 보냄 이는 메세지 손실을 방지하기 위해 사용함 | lhy | 1121
			auto_ack = True
		)
		logger.info('main is end...')
		
		return 
		
	def on_message(channel, method_frame, header_frame, body):
		
		logger.info('on_message method start')

		# message 생성
		result = {}
		param = {}
		trainResult = {}
		logger.info('on_message body start')
		logger.info(body)
		# ack 수동 ***(queue ack)***
		# channel.basic_ack(delivery_tag=method_frame.delivery_tag)
		message = str(body, "utf-8")
		try:
			if message != '':
				logger.info('on_message param start')
				# 학습 전에 body message(str)를 dict로 변환
				try:
					param = json.loads(message, strict=False)
					logger.info('on_message param end')
				except Exception as e:
					logger.info(e)
					result['callbackUrl'] = param['callbackUrl']
					result['trainInstanceId'] = param['trainInstanceId']
					result['trainExperimentId'] = param['trainExperimentId']
					result['resultCode'] = '2000'
					result['result'] = '학습 실행 실패 : ' + str(e)
				try:
					logger.info('train run start')
					logger.info(param)

# 					# minio
					path = '/minio/BaseEngine'
					bucketName = param['baseModelNm']
					version = param['baseModelVersion']

					# 임시로 version 제외
					# save_path = os.path.join(path, bucketName, version)
					save_path = os.path.join(path, bucketName)
					# createDirectory(save_path)
					if os.path.exists(save_path) :
						shutil.rmtree(save_path)
						os.makedirs(save_path)
					else:
						os.makedirs(save_path)

					client = AsyncioRabbitMQTrain.client

					# files = client.list_objects(bucketName, prefix = version, use_url_encoding_type = 'utf-8', recursive = True)
					files = client.list_objects(bucketName, use_url_encoding_type = 'utf-8', recursive = True)
					for _file in files :
						out_path = os.path.join(path, bucketName, _file.object_name)
						logger.info("out_path")
						logger.info(out_path)
						client.fget_object(bucketName, _file.object_name, out_path)
						filename = _file	

					_list = os.listdir(save_path)
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
					
					
					 # 자동분할 성능테스트가 있는경우 프로세스종료하지 않음
					for i in param['trainConfig']['params']:
						if i['param_name'] == "split_ratio":
							if i['param_value'][2] != 0:
								AsyncioRabbitMQTrain.autoTest = True
					logger.info("AsyncioRabbitMQTrain.autoTest")
					logger.info(AsyncioRabbitMQTrain.autoTest)
					# baseengine_path='/nlu/BaseEngine/T5/kt-ulm-3b-no9'
					# 모델 학습 
					# 사용량 데이터를 저장  param['requestId'] 추가 | lhy | 1222
					experiment = SummarizationTrainPipeline(param['trainDataUrl'], str(param['trainConfig']), param['fineTunningPath'], param['requestID'], baseengine_path, bucketName)
					trainResult = experiment()

					# shutil.rmtree(save_path)

					if trainResult['resultCode'] == 0:
						# 모델 학습이 정상적으로 종료됐을 경우 deliveries 확인
						AsyncioRabbitMQTrain._message_number += 1
						AsyncioRabbitMQTrain._deliveries.append(AsyncioRabbitMQTrain._message_number)
						logger.info('Published message # %i', AsyncioRabbitMQTrain._message_number)
						
						# mq message 작성
						result['callbackUrl'] = param['callbackUrl']
						result['trainInstanceId'] = param['trainInstanceId']
						result['trainExperimentId'] = param['trainExperimentId']
						result['resultCode'] = '0000'
						result['result'] = '정상 응답'
						result["resultReport"] = trainResult['resultReport']
					
					#0210 추가
					else:
						result['callbackUrl'] = param['callbackUrl']
						result['trainInstanceId'] = param['trainInstanceId']
						result['trainExperimentId'] = param['trainExperimentId']
						result['resultCode'] = '2000'
						result['result'] = '학습 실행 실패 : ' + str(e)
						# result["resultReport"] = trainResult['resultReport']

				except Exception as e:
					logger.info(e)
					result['callbackUrl'] = param['callbackUrl']
					result['trainInstanceId'] = param['trainInstanceId']
					result['trainExperimentId'] = param['trainExperimentId']
					result['resultCode'] = '2000'
					result['result'] = str(e)

			else:	
				result['callbackUrl'] = param['callbackUrl']
				result['trainInstanceId'] = param['trainInstanceId']
				result['trainExperimentId'] = param['trainExperimentId']
				result['resultCode'] = '2000'
				result['result'] = '학습 실행 실패 : ' + str(e)

		except Exception as e:
			logger.info(e)
			result['callbackUrl'] = param['callbackUrl']
			result['trainInstanceId'] = param['trainInstanceId']
			result['trainExperimentId'] = param['trainExperimentId']
			result['resultCode'] = '2000'
			result['result'] = '학습 실행 실패 : ' + str(e)
		
		finally:
			
			message = ''
			message = json.dumps(result)
			logger.info('message send start')
			
			# train run 완료 후 TrainResult Queue에 message 
			logger.info(AsyncioRabbitMQTrain.EXCHANGE)
			logger.info(AsyncioRabbitMQTrain.ROUTING_KEY)
			logger.info(result)
			
			logger.info('basic_publish run start')
			channel.basic_publish(AsyncioRabbitMQTrain.EXCHANGE, AsyncioRabbitMQTrain.ROUTING_KEY, message)
			logger.info('basic_publish run end')
			
			return
