# Collection of generic functions that any TensorFlow-Keras model-based method can use
import tensorflow as tf
from tensorflow.keras.models import load_model

import os

def initDevice(runGPU):
	"""
	If running on GPU, limit GPU memory growth
	If running on CPU, hide any GPUs from TF
	"""

	if runGPU:
		# make sure TF doesn't gobble up device memory
		gpus = tf.config.experimental.list_physical_devices('GPU')
		if gpus:
			try:
				# Currently, memory growth needs to be the same across GPUs
				for gpu in gpus:
					tf.config.experimental.set_memory_growth(gpu, True)
					logical_gpus = tf.config.experimental.list_logical_devices('GPU')
			except RuntimeError as e:
				# Memory growth must be set before GPUs have been initialized
				print(e)
	else:
		# If GPU is available, TF will automatically run there
		#	This forces to run on CPU even if GPU is available
		os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 


def loadModelObj(modelPath):
	"""
	Load Keras SavedModel object from file specified by modelPath
	"""

	modelObj = load_model(modelPath, compile=False)
	return modelObj


def getIOShape(shape):
	"""
	Gets model I/O shape, handles situations in which layer shape is returned as a list instead of a tuple
	"""

	if type(shape) is list:
		if (len(shape) != 1):
			raise ValueError("Invalid TF model I/O size")
		else:
			shape = shape[0]

	return shape