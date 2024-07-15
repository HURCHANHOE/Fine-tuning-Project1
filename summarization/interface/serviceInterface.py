from abc import ABC, abstractmethod
import os
import sys
from queue import Queue

class servicePipelineInterface(ABC):
	
	@abstractmethod
	def __init__(self):
		pass
		
	@abstractmethod
	def load_model(self):
		pass

	@abstractmethod
	def inference(self):
		pass


	
