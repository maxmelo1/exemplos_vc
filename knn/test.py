import numpy as np
import matplotlib.image as mpimg
import os
from imutils import paths
from skimage.color import rgb2gray
import skimage.feature as ft
import csv



class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        """ X é NxD onde cada linha é um exemplo que queremos predizer o label"""
        num_test = X.shape[0]

        print("%d amostras"%(X.shape[0]) )

        #criando o tipo
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)


        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis = 1)
            min_index = np.argmin(distances) #pegar o menor
            Ypred[i]  = self.ytr[min_index] #predizer o label do exemplo mais próximo

        return Ypred


def main():

    trainImgs   = []
    testImgs    = []

    trainLabels = []
    testLabels  = []

    with open( '../gencsv/lbp_train.csv', mode='r' ) as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            trainImgs.append( row[0] )
            trainLabels.append( row[1] )
            print(row[0])

    with open( '../gencsv/lbp_test.csv', mode='r' ) as file:
        reader = csv.reader(file, delimiter=',')

        for row in reader:
            testImgs.append( row[0] )
            testLabels.append( row[1] )


    cl = NearestNeighbor()

    test = np.array(trainImgs, dtype=np.int64)

    cl.train( np.array(trainImgs, dtype=np.int64), np.array(trainLabels, dtype=np.int64) )

    y = cl.predict( np.array(testImgs, dtype=int) )

    acc = 0

    print(len(y))

    for i in range(len(y)):
        acc = acc + 1 if y[i] == testLabels[i] else acc

    print( "Acertos: %d de %d, %f \%" % (acc, len(y), acc/len(y) ) )


if __name__ == '__main__':
    main()
