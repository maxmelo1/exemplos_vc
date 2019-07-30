from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.preprocessing.image import ImageDataGenerator
from IPython.display import display
from PIL import Image
from keras.preprocessing import image
import numpy as np
import os
#from keras.models import model_from_json
from keras.models import load_model


class ClassifyDogsAndCats:
    def __init__(self):
        self.classifier = Sequential()

    def train(self):
        #conv 1
        self.classifier.add( Convolution2D(32, 3,3, input_shape = (64,64,3) , activation = 'relu') )

        #pooling 1
        self.classifier.add(MaxPooling2D(pool_size = (2,2)))

        #conv 2
        self.classifier.add( Convolution2D(64, 3,3, input_shape = (64,64,3), activation = 'relu') )

        #pooling 2
        self.classifier.add(MaxPooling2D(pool_size = (2,2)))

        #flatten 2
        self.classifier.add(Flatten())

        #fc
        self.classifier.add( Dense(output_dim = 128, activation = 'relu') )
        self.classifier.add( Dense(output_dim = 1, activation = 'sigmoid') )

        #compiling
        self.classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    
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
        self.classifier.fit_generator(
            self.trainingSet,
            steps_per_epoch = 8000,
            epochs = 15,
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
        
        print("Saved model to disk")

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

        print("Loaded model from disk")

        #compiling
        #self.classifier.compile( optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'] )
    
    def test(self, imageName):
        
        testImage = image.load_img(imageName, target_size = (64,64))
        testImage = image.img_to_array(testImage)
        testImage = np.expand_dims(testImage, axis = 0)

        result = self.classifier.predict(testImage)

        print("details")
        print(result)

        if result[0][0] > 0.5:
            return "dog"
        return "cat"

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
        obj.train()
        #2
        obj.augmentDataset()
        #3
        obj.fit()
        #4
    elif menu == 2:
        obj.loadClassifier()
    
    if menu > 0:
        for i in range(1,18):
            print( obj.test("tests/test"+str(i)+".jpeg" ) ) 



if __name__ == '__main__':
    main()