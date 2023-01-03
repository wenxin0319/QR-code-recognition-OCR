from math import e
import pickle
import numpy as np
from PIL import Image
import numpy as np 
import os
from OCRCommon import MODEL_WEIGHTS_FILENAME, NAME_TO_CLASS, NUM_CLASS, VC_HEIGHT, VC_OCR_CHINNELS, VC_WIDTH
from VC4OCRNet import VC4OCRNet

def load_train_data(directory, max_num = 1000):
    filenames = os.listdir(directory)
    numfiles = len(filenames)
    trainX = np.ndarray(shape=(min([numfiles, max_num]), VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS), dtype='float32')
    trainY = np.zeros(shape=(min([numfiles, max_num]), 4, NUM_CLASS), dtype='float32')

    Xid = 0
    for fname in filenames:
        y = fname.split('_')[1].split('.')[0]
        img_obj = Image.open(os.path.join(directory, fname))
        img_gray = img_obj.convert('L')
        trainX[Xid, :, :, :] = np.array(img_gray).astype('float32').reshape(VC_HEIGHT, VC_HEIGHT, VC_OCR_CHINNELS) / 255.0
        img_gray.close()
        img_obj.close()
        trainY[Xid, 0, NAME_TO_CLASS[y[0]]] = 1.0
        trainY[Xid, 1, NAME_TO_CLASS[y[1]]] = 1.0
        trainY[Xid, 2, NAME_TO_CLASS[y[2]]] = 1.0
        trainY[Xid, 3, NAME_TO_CLASS[y[3]]] = 1.0

        Xid += 1
        if Xid == max_num:
            print("Stop load at Xid={}".format(Xid))
            break

    return trainX, trainY

def load_train_data_packed(directory):
    xfile = os.path.join(directory, "packed_dataX.npy")
    yfile = os.path.join(directory, "packed_dataY_text_label.pickle")
    trainX = np.load(xfile)
    trainY = np.zeros(shape=(trainX.shape[0], 4, NUM_CLASS))
    labels = []
    with open(yfile, "rb") as f:
        labels = pickle.load(f)
    Xid = 0
    for y in labels:
        trainY[Xid, 0, NAME_TO_CLASS[y[0]]] = 1.0
        trainY[Xid, 1, NAME_TO_CLASS[y[1]]] = 1.0
        trainY[Xid, 2, NAME_TO_CLASS[y[2]]] = 1.0
        trainY[Xid, 3, NAME_TO_CLASS[y[3]]] = 1.0
        Xid += 1
    return trainX, trainY


if __name__ == "__main__":
    trainX, trainY = load_train_data_packed("./validation_code_2_packed")
    print("Shape of trainX = {}".format(trainX.shape))
    print("Shape of trainY = {}".format(trainY.shape))
    
    model = VC4OCRNet()
    #if not os.path.exists("VC4OCRNet"):
    model.fit(trainX, trainY, epochs=12)
    if os.path.exists("./VC4OCRNet.h5"):
        os.remove("./VC4OCRNet.h5")

    model.save(MODEL_WEIGHTS_FILENAME)
    #else:
    #    model.load("VC4OCRNet")
