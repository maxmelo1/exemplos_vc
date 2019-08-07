from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten, Dropout, Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
from keras.preprocessing import image
import numpy as np
import os
#from keras.models import model_from_json
from keras.models import load_model
from keras.regularizers import l2
from keras import backend as K
import matplotlib.pyplot as plt
from keras.utils import plot_model
import pydot as pyd



class ClassifyDogsAndCats:
    def __init__(self):
        self.classifier = Sequential()

    def train(self, init, reg):
        #block 1
        #self.classifier.add( Convolution2D(32, (3, 3), input_shape = (64,64,3), strides=(1, 1), kernel_initializer=init, kernel_regularizer=reg, activation = 'relu') )
        self.classifier.add( Convolution2D(32, (3, 3), input_shape = (64,64,3), strides=(1, 1), activation = 'relu') )
        self.classifier.add( Convolution2D(32, (3, 3), input_shape = (64,64,3), strides=(1, 1), activation = 'relu') )
        self.classifier.add(MaxPooling2D(pool_size = (2,2)))
        self.classifier.add(Dropout(0.25))

        #block 2
        #self.classifier.add( Convolution2D(64, (3, 3), input_shape = (64,64,3), strides=(2, 2) , activation = 'relu') )
        #self.classifier.add( Convolution2D(64, (3, 3), input_shape = (64,64,3), strides=(2, 2), activation = 'relu') )
        #self.classifier.add(MaxPooling2D(pool_size = (2,2)))
        #self.classifier.add(Dropout(0.25))

        #flatten
        self.classifier.add(Flatten())
        #fc
        self.classifier.add(Dense(output_dim = 128, activation='relu'))
        #self.classifier.add(Dropout(0.5))
        self.classifier.add(Dense(output_dim = 1, activation='sigmoid'))
        #self.classifier.add(Dense(output_dim = 1, activation='softmax'))

        #compiling
        self.classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
        #self.classifier.compile( optimizer = 'rmsprop', loss = 'binary_crossentropy', metrics = ['accuracy'] )

    def augmentDataset(self):
        self.trainDatagen = ImageDataGenerator(
            rescale         = 1./255,
            shear_range     = 0.2,
            zoom_range      = 0.2,
            horizontal_flip = True
        )

        self.testDatagen = ImageDataGenerator(rescale=1./255)

        self.trainingSet = self.trainDatagen.flow_from_directory(
            'dataset/training_set',
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'binary'
        )

        self.testSet = self.testDatagen.flow_from_directory(
            'dataset/test_set',
            target_size = (64, 64),
            batch_size = 32,
            class_mode = 'binary'
        )

    def fit(self):
        self.history = self.classifier.fit_generator(
            self.trainingSet,
            steps_per_epoch = 40,
            epochs = 2,
            validation_data = self.testSet,
            validation_steps = 800,
            use_multiprocessing=True,
            workers=4
        )

        # serialize model to JSON
        #model_json = self.classifier.to_json()
        #with open("model.json", "w") as json_file:
        #    json_file.write(model_json)
        # serialize weights to HDF5
        #self.classifier.save_weights("model.h5")

        self.classifier.save("test_model.h5")

        print("Model saved to disk")

        #print(self.history.history.keys())

        self.plotHistory()

    def plotHistory(self):
        plt.plot(self.history.history['acc'])
        plt.plot(self.history.history['val_acc'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        #summarize history for loss
        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        plt.show()

        plot_model(self.classifier, to_file='model.png')


    def loadClassifier(self):
        # load json and create model
        #json_file = open('model.json', 'r')
        #loaded_model_json = json_file.read()
        #json_file.close()
        #loaded_model = model_from_json(loaded_model_json)
        # load weights into new model
        #loaded_model.load_weights("model.h5")

        #self.classifier = loaded_model

        self.classifier = load_model("test_model.h5")

        print("Model loaded from disk")

        #compiling
        #self.classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )

    def test(self, imageName):

        testImage = image.load_img(imageName, target_size = (64,64))
        testImage = image.img_to_array(testImage)
        testImage = np.expand_dims(testImage, axis = 0)

        result = self.classifier.predict(testImage)

        #print(result)

        if result[0][0] > 0.5:
            return "dog"
        return "cat"

    def showHeatMap(self):
        # Grad-CAM algorithm
        specoutput = self.classifier.output[:, 668]
        last_conv_layer = self.classifier.get_layer('block5_conv3')
        grads = K.gradients(specoutput, last_conv_layer.output)[0]
        pooled_grads = K.mean(grads, axis=(0, 1, 2))
        iterate = K.function([self.classifier.input], [pooled_grads, last_conv_layer.output[0]])
        pooled_grads_value, conv_layer_output_value = iterate([image])
        for i in range(512):
            conv_layer_output_value[:, :, i] *= pooled_grads_value[i]
            heatmap=np.mean(conv_layer_output_value, axis=-1) # Heatmap post processing
        heatmap = np.maximum(heatmap, 0)
        heatmap /= np.max(heatmap)
        plt.matshow(heatmap)
        plt.show()

def main():
    obj = ClassifyDogsAndCats()

    menu = -1
    while( menu < 0 or menu > 2 ):
        print("Uso:")
        print("0 - sair")
        print("1 - treinar a CNN")
        print("2 - carregar CNN do arquivo")

        menu = int(input())


    if( menu == 1 ):
        #1
        obj.train(init="he_normal", reg=l2(0.0005))
        #2
        obj.augmentDataset()
        #3
        obj.fit()
        #4
    elif menu == 2:
        obj.loadClassifier()

    if menu > 0:
        template = ["dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","dog","cat","cat","dog",]
        size = len(template)

        acc = 0

        print("details")

        for i in range(1,size):
            ans = obj.test("tests/test"+str(i)+".jpeg" )
            result = "wrong"
            if(template[i] == ans):
                result = "right"
                acc = acc + 1
            print( "predicted %s - template says %s - %s" %(ans, template[i], result ) )

        print("%.2f accuracy" %(acc /18))


if __name__ == '__main__':
    main()
