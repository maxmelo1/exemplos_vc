from imutils import paths
from skimage.color import rgb2gray
import matplotlib.image as mpimg
import skimage.feature as ft
import csv

class GenCSV:
    # settings for LBP
    METHOD = 'uniform'
    P = 16
    R = 2

    def __init__(self):
        pass

    def generateLBP(self, csvName, path):
        trainPaths = list(paths.list_images(path))

        trainLabels = []
        trainImgs   = []

        with open( csvName, mode='w') as file:
            writer = csv.writer(file, delimiter = ',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

            for path in trainPaths:
                str = path.split("/")
                label =  0 if "dog" in str[2] else 1

                img  = rgb2gray(mpimg.imread(path))
                tere = ft.local_binary_pattern(img, self.P, self.R, self.METHOD)

                writer.writerow([tere, label])

def main():
    obj = GenCSV()
    obj.generateLBP('lbp_train.csv', "../knn/dogs-vs-cats/train")
    obj.generateLBP('lbp_test.csv', "../knn/dogs-vs-cats/test1")



if __name__ == '__main__':
    main()
