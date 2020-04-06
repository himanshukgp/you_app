from scripts.mtcnn import mtcnn
from matplotlib import pyplot
from scripts.facenet import faceNet



def main():
	filename = 'f3.jpeg'
	database = 'data'
	image = pyplot.imread(filename)
	model = mtcnn()
	model.evaluate(image)
	#model.draw_faces()
	model.get_faces()
	face = faceNet(database, model.faces)
	face.evaluate()




if __name__ == "__main__":
    main()