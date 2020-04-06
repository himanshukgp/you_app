from tensorflow.keras.models import load_model
from keras_facenet import FaceNet
from PIL import Image
import numpy as np
import os
import cv2

class faceNet(object):
	"""docstring for FaceNet"""
	def __init__(self, path_database, image):
		
		self.embedder = FaceNet()
		self.image = image
		self.path_database = path_database
		self.embedding_database = None
		self.embedding_group = None
		self.names = None

	def embedd(self, arr):
		embeddings = self.embedder.embeddings(arr)
		return embeddings

	def convert_database(self, image, required_size=(160, 160)):
		image = Image.fromarray(image)
		image = image.convert('RGB')
		image = image.resize(required_size)
		face_array = np.asarray(image)
		return face_array

	def load_images_from_folder(self):
		images = []
		names = []
		for filename in os.listdir(self.path_database):
			names.append(filename.split(".")[0])
			img = cv2.imread(os.path.join(self.path_database, filename))
			if img is not None:
				images.append(img)
		return images, names


	def evaluate(self):
		images, names = self.load_images_from_folder()
		self.names = names
		database = []
		for i in range(len(images)):
			database.append(self.convert_database(images[i]))

		self.embedding_database = self.embedd(np.array(database))
		self.embedding_group = self.embedd(np.array(self.image))
		self.result()

	def result(self):
		for i in range(len(self.embedding_database)):
			for j in range(len(self.embedding_group)):
				dist = np.linalg.norm(self.embedding_database[i] - self.embedding_group[j])
				if(dist<1):
					print(self.names[i])
					break



		
