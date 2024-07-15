import os, sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname((__file__)))))))
import logging
import pika
from pika.adapters.asyncio_connection import AsyncioConnection
import json
from train.summarization_train_pipeline import SummarizationTrainPipeline
from common import pidKill
from common import get_rouge

# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))

from service.summarization_service_pipeline import SummarizationServicePipeline

FM = SummarizationServicePipeline()

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(message)s', datefmt='%Y-%m-%dT%H:%M:%S%z')
logger = logging.getLogger(__name__)

instanceid = ''
instanceid = os.environ['instanceid']

testQueueName = 'Test_' + instanceid
testRoutingKey = 'TestResult.all.test'

rabbitMqUrl = os.environ['rabbitMqUrl']
rabbitMqVhost = os.environ['rabbitMqVhost']
rabbitMqPort = os.environ['rabbitMqPort']
rabbitMqId = os.environ['rabbitMqId']
rabbitMqPw = os.environ['rabbitMqPw']


class AsyncioRabbitMQTest(object):
	EXCHANGE = 'nlu-topic-exchange'
	# EXCHANGE_TYPE = ExchangeType.topic
	# PUBLISH_INTERVAL = 1
	QUEUE = testQueueName
	ROUTING_KEY = testRoutingKey
	
	_message_number = 0
	_deliveries = []
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
		if len(AsyncioRabbitMQTest._deliveries) > 0:
			AsyncioRabbitMQTest._deliveries.remove(method_frame.method.delivery_tag)
		logger.info(
			'Published %i messages, %i have yet to be confirmed, '
			'%i were acked and %i were nacked', AsyncioRabbitMQTest._message_number,
			len(AsyncioRabbitMQTest._deliveries), self._acked, self._nacked)
		# 메모리 누수 문제를 임시적으로 해결하는 방편 pid를 확인하여 kill | lhy | 1220
		pidKill()
		
	def main(self):
		logger.info('main is starting...self.QUEUE is : ')
		logger.info(self.QUEUE)

		self._channel.basic_consume(
			queue = self.QUEUE, 
			on_message_callback = AsyncioRabbitMQTest.on_message,
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
			testData = {}
			testResult = {}
			utteranceList = []
			summmarList=[]
			summmarList_Label = []
			model = None
			tokenizer = None
			save_path = None
			# base_path = '/minio/BaseEngine/kt-midm-11b'
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
						try:
							# fineTunningModel load하여 inference 준비
							logger.info('load_model start')
							model, tokenizer = FM.load_model(param['fineTunningPath'], save_path)
							logger.info('load_model end')

							# test utterance json load
							with open(param['testDataUrl'], 'r', encoding='utf-8') as file:
								testData = json.load(file)
								
							utteranceList = testData['utterances']
							
							if len(utteranceList) > 0:
								#utterance의 intents 값을 추론 
								testResultList = []
								correctCount = 0
								acc = 0
								_sum = 0
								cnt = 0
								accuracy = ''
                                # 성능테스트데이터 전체 토큰수 20230608 lhy
								totalToken = 0
                                
								# utteranceList 길이만큼 반복문 수행
								for utterence in utteranceList:
									intentList = {}
									# 추론서비스틀 이용하여 label 추론
									intentList = FM.inference(model, tokenizer, utterence['utterance'])
                                    
									# logger.info("intentList")
									# logger.info(intentList)
                                    
									logger.info(intentList["result"][0]['result'][0])
									
									logger.info("utterence['correctIntent']")
									logger.info(utterence['correctIntent'])
                                    
									#예측 데이터
									tokenized_preds = [tokenizer.tokenize(intentList["result"][0]['result'][0])]
									# 정답 데이터
									tokenized_labels = [tokenizer.tokenize(utterence['correctIntent'])]
									# 성능테스트데이터 전체 토큰수 20230608 lhy
									totalToken  = totalToken + intentList['total_token_count']
                                    
									# confidence
									rouge_score = get_rouge(tokenized_preds, tokenized_labels)
									logger.info("rouge_score")
									logger.info(rouge_score)
									
									rouge_score_dict = {
										"result" : intentList["result"][0]["result"][0],
										"confidence" : round(rouge_score, 3)
									}
									
									intentList["result"] = [rouge_score_dict]
									testResultTemp = intentList["result"]
									utterence['intents'] = testResultTemp
									
									logger.info("utterence")
									logger.info(utterence)
									# summmarList.append(rouge_score)
									# correctIntent와 추론된 Intent가 같으면 정답
								
									_sum += rouge_score
									cnt+=1
								
								testResult['utterances'] = utteranceList
								# f1 score 평균낸거
								testResult['accuracy'] = round(_sum/cnt, 4)
								# 성능테스트데이터 토큰수 저장
								testResult['totalToken'] = totalToken
								# 모델 학습 report.json 저장과 같은 방식 | 성능테스트 사용량 저장 key값 20230608 lhy 
								# api 배포시 주석 해제 필요
								# testResult['requestId'] = param['requestId']
                                
								testCode = '0'

							else : 
								testCode = '1'

							# directory 생성을 위해 파일명 분리
							path, files = os.path.split(param['testReportUrl'])

							if testCode == '0':
								if not os.path.exists(path): 
									os.makedirs(path)

								# testReportUrl에 testResult를 json으로 저장
								with open(param['testReportUrl'], 'w', encoding="utf-8") as file:
									json.dump(testResult, file, ensure_ascii=False, indent="\t")

								# callback 
								result['callbackUrl'] = param['callbackUrl']
								result['trainInstanceId'] = param['trainInstanceId']
								result['trainExperimentId'] = param['trainExperimentId']
								result['modelTestId'] = param['modelTestId']
								result['resultCode'] = '0000'
								result['result'] = '정상 응답'
								result["accuracy"] = testResult['accuracy']
								result["testReportUrl"] = param['testReportUrl']

						except Exception as e:
							logger.info(e)							
							result['callbackUrl'] = param['callbackUrl']
							result['trainInstanceId'] = param['trainInstanceId']
							result['trainExperimentId'] = param['trainExperimentId']
							result['modelTestId'] = param['modelTestId']
							result['resultCode'] = '8888'
							result['result'] = '성능 테스트 실행 실패 : ' + str(e)
					except Exception as e:
						logger.info(e)
						result['callbackUrl'] = param['callbackUrl']
						result['trainInstanceId'] = param['trainInstanceId']
						result['trainExperimentId'] = param['trainExperimentId']
						result['modelTestId'] = param['modelTestId']
						result['resultCode'] = '8888'
						result['result'] = '성능 테스트 실행 실패 : ' + str(e)
					
			except Exception as e:
				logger.info(e)
				result['callbackUrl'] = param['callbackUrl']
				result['trainInstanceId'] = param['trainInstanceId']
				result['trainExperimentId'] = param['trainExperimentId']
				result['modelTestId'] = param['modelTestId']
				result['resultCode'] = '8888'
				result['result'] = '성능 테스트 실행 실패 : ' + str(e)
			
			finally:
							
				message = ''
				message = json.dumps(result)
				logger.info('message send start')
				
				# train run 완료 후 TrainResult Queue에 message 
				logger.info(AsyncioRabbitMQTest.EXCHANGE)
				logger.info(AsyncioRabbitMQTest.ROUTING_KEY)
				logger.info(result)
				
				logger.info('basic_publish run start')
				# routing key, exchange로 messgae binding 
				channel.basic_publish(AsyncioRabbitMQTest.EXCHANGE, AsyncioRabbitMQTest.ROUTING_KEY, message)
				logger.info('basic_publish run end')			
				return
