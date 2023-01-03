from sklearn.metrics import accuracy_score
#from tensorflow import keras
import tensorflow 
if int(tensorflow.__version__[0]) == 2:
    tf1 = tensorflow.compat.v1
    tf1.disable_eager_execution()
    import tensorflow as tf2
    from tensorflow import keras
else:
    import tensorflow as tf1

import numpy as np
from PIL import Image
import numpy as np 
from OCRCommon import NUM_CLASS, VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS

class VC4OCRNet:
        
    def __init__(self):
        self.input = keras.Input(shape=(VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS))

        x = (keras.layers.Conv2D(filters=32, kernel_size=(3, 3)))(self.input)
        x = (keras.layers.PReLU())(x)
        x = (keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))(x)

        x = (keras.layers.Conv2D(filters=64, kernel_size=(5, 5)))(x)
        x = (keras.layers.PReLU())(x)
        x = (keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))(x)
        
        x = (keras.layers.Conv2D(filters=128, kernel_size=(5, 5)))(x)
        x = (keras.layers.PReLU())(x)
        x = (keras.layers.MaxPool2D(pool_size=(2,2), strides=2, padding='same'))(x)

        x = (keras.layers.Conv2D(filters=256, kernel_size=(2, 2)))(x)
        x = (keras.layers.PReLU())(x)
        #x = (keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, padding='same'))(x)

        #x = (keras.layers.Flatten())(x)
        #x = (keras.layers.Dense(units=256))(x)
        #x = (keras.layers.PReLU())(x)

        x = (keras.layers.Flatten())(x)
        x = (keras.layers.Dense(units=4 * NUM_CLASS))(x)
        x = (keras.layers.PReLU())(x)

        x = (keras.layers.Reshape([4, NUM_CLASS]))(x)
        x = (keras.layers.Softmax())(x)

        self.model : keras.Model = keras.Model(inputs=self.input, outputs=x, name='CRNN')
        self.model.compile(optimizer=keras.optimizers.Adamax(), 
                                loss=keras.losses.categorical_crossentropy, 
                             metrics=['accuracy'])
        
    
    def fit(self, trainX, trainY, epochs=22, batch_size=75):
        self.model.fit(trainX, trainY, epochs=epochs, batch_size=batch_size, shuffle=False)

    def save(self, path):
        self.model.save_weights("./" + path)

    def load(self, path):
        self.model.load_weights("./" + path)
        #self.model = keras.models.load_model("./" + path)

    def predict(self, X):
        return self.model.predict(X)
        
if __name__ == "__main__":
    VC4 = VC4OCRNet()