import numpy as np
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot
from tqdm import tqdm

class mtcnn():

	def __init__(self):
		self.model = MTCNN()
		self.boxes = None
		self.faces = None
		self.image = None

	def evaluate(self, image=None):
		self.image = image
		if image is not None:
			self.boxes = self.model.detect_faces(image)

	def get_one_face(self, faces, required_size=(160, 160)):
		x1, y1, width, height = faces['box']
		x1, y1 = abs(x1), abs(y1)
		x2, y2 = x1 + width, y1 + height
		face = self.image[y1:y2, x1:x2]
		image = Image.fromarray(face)
		image = image.resize(required_size)
		face_array = np.asarray(image)
		return face_array

	def get_faces(self):
		if (self.boxes is not None):
			face_array = []
			for i in tqdm(range(len(self.boxes))):
				one_face = self.get_one_face(self.boxes[i])
				face_array.append(one_face)
			self.faces = np.array(face_array)

	def draw_faces(self):
		data = self.image
		result_list = self.boxes
		for i in tqdm(range(len(result_list))):
			x1, y1, width, height = result_list[i]['box']
			x2, y2 = x1 + width, y1 + height
			pyplot.subplot(1, len(result_list), i+1)
			pyplot.axis('off')
			pyplot.imshow(data[y1:y2, x1:x2])
		pyplot.show()
