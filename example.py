from scripts.mtcnn import mtcnn
from matplotlib import pyplot

def main():
	filename = 'f3.jpeg'
	image = pyplot.imread(filename)
	model = mtcnn()
	model.evaluate(image)
	model.draw_faces()


if __name__ == "__main__":
    main()