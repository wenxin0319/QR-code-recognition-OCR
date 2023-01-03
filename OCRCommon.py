CLASS_TO_NAME = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
                'A', 'B', 'C', 'D', 'E', 'F', 'G', 
                'H', 'I', 'J', 'K', 'L', 'M', 'N', 
                'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

NUM_CLASS = len(CLASS_TO_NAME)
NAME_TO_CLASS = dict()
for i in range(len(CLASS_TO_NAME)):
    NAME_TO_CLASS[CLASS_TO_NAME[i]] = i

TRAIN_KEEP_PROB = 0.8
TEST_KEEP_PROB = 1.0
VC_WIDTH = 100
VC_HEIGHT = 40
VC_OCR_CHINNELS = 3

MODEL_WEIGHTS_FILENAME = "VC4OCRNet.h5"