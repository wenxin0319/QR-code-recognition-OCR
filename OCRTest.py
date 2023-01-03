from numpy.core.fromnumeric import shape
from OCRTrain import load_train_data_packed
from VC4OCRNet import VC4OCRNet
from OCRCommon import MODEL_WEIGHTS_FILENAME, CLASS_TO_NAME
import numpy as np
from tensorflow import keras

def accuracy_each_char(y_true, y_pred):
    return np.sum((y_true.reshape(-1) == y_pred.reshape(-1)).astype('int')) / (y_true.shape[0] * y_true.shape[1]) 

def accuracy_each_vc4(y_true, y_pred):
    return np.sum(np.sum(((y_true == y_pred).astype('int') / y_true.shape[1]), axis=1).astype('int')) / y_true.shape[0]

if __name__ == "__main__":
    ocrnet = VC4OCRNet()
    ocrnet.load(MODEL_WEIGHTS_FILENAME)
    testX, testY = load_train_data_packed("test_data")
    predY = ocrnet.predict(testX)
    
    test_labels = np.argmax(testY, axis=2)
    pred_labels = np.argmax(predY, axis=2)
    
    print("Accuracy for each char: {}".format(accuracy_each_char(test_labels, pred_labels)))
    print("Accuracy for each vc4 : {}".format(accuracy_each_vc4(test_labels, pred_labels)))