from abc import ABC, abstractmethod
import os
import sys

#traininterface
class trainPipelineInterface(ABC):

	@abstractmethod
	def __init__(self):
		pass
	   
	@abstractmethod
	def prepare_data(self):
		pass

	@abstractmethod
	def prepare_model(self):
		pass

	@abstractmethod
	def do_train(self):
		pass

	@abstractmethod
	def save_model(self):
		pass
