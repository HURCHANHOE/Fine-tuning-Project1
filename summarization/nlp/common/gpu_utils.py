import torch
class KTGpuUtils:
	def __init__(self):
		self.gpu_count=torch.cuda.device_count()
		self.gpu_mem_info={}
		self.gpu_device_info={}
		for gpu_id in range(self.gpu_count):
			self.gpu_mem_info[gpu_id]={'total':0,'available':0,'used':0}
			self.gpu_device_info[gpu_id]=torch.cuda.get_device_properties(gpu_id)
		self.get_available_memory()

	def get_available_memory(self):
		for gpu_id in range(self.gpu_count):
			torch.cuda.set_device(gpu_id)
			(available,max)=torch.cuda.mem_get_info()
			used=max-available
			self.gpu_mem_info[gpu_id]['total']=max/1024/1024
			self.gpu_mem_info[gpu_id]['available']=available/1024/1024
			self.gpu_mem_info[gpu_id]['used']=used/1024/1024
		return (self.gpu_count,self.gpu_mem_info)

	def get_max_available_mem_device(self):
		gpu_cnt,mem_info=self.get_available_memory()
		return_gpu_id=0
		return_mem_available=0
		for gpu_id in range(gpu_cnt):
			if mem_info[gpu_id]['available']>return_mem_available:
				return_gpu_id=gpu_id
				return_mem_available=mem_info[gpu_id]['available']
		return (return_gpu_id,return_mem_available)
		
	def get_gpu_info(self,gpu_id):
		return {'name':self.gpu_device_info[gpu_id].name,'total_memory':self.gpu_device_info[gpu_id].total_memory/1024/1024}
