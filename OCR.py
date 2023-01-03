from VC4OCRNet import VC4OCRNet
import sys 
import numpy as np
from OCRCommon import MODEL_WEIGHTS_FILENAME, VC_WIDTH, VC_HEIGHT, VC_OCR_CHINNELS, CLASS_TO_NAME
from PIL import Image
from matplotlib import pyplot as plt

if __name__ == "__main__":
    ocrnet = VC4OCRNet()
    ocrnet.load(MODEL_WEIGHTS_FILENAME)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
        X = np.ndarray(shape=(1, VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS))
        img = None
        if VC_OCR_CHINNELS == 3:
            img = Image.open(filename).convert('RGB') 
        elif VC_OCR_CHINNELS == 1:
            img = Image.open(filename).convert('L')
        else:
            print("Unsupported format")
            exit(0)
        X[0, :, :, :] = (np.array(img).astype('float32')).reshape(VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS) / 255.0
        img.close()
        Y = ocrnet.predict(X)[0]
        ans = ''.join([CLASS_TO_NAME[c] for c in np.argmax(Y, axis=1)])
        print(ans)
        plt.figure()
        plt.imshow(X[0, :, :, :])
        plt.title("OCR predicts: {}".format(ans))
        plt.show()
