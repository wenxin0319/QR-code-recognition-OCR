from PIL import Image, ImageDraw, ImageFont
from OCRCommon import VC_WIDTH, VC_HEIGHT, CLASS_TO_NAME, VC_OCR_CHINNELS
import numpy as np 
import random
import math
import sys
import pickle

FontNames = [("msyh.ttc"), ("arial.ttf"), ("simsunb.ttf"), ("simsun.ttc"), ("simhei.ttf"),
         ("STZHONGS.TTF"), ("SIMYOU.TTF"), ("FZYTK.TTF"), ("Deng.ttf"),
         ("STENCIL.TTF"), ("tahoma.ttf"), ("times.ttf"), ("sylfaen.ttf"),
         ("POORICH.TTF"), ("MAIAN.TTF"), ("georgiai.ttf"), ("cambria.ttc"), 
         ("AGENCYR.TTF"), ("BOOKOSB.TTF"), ("BOOKOS.TTF"), ("seguisb.ttf"),
         ("calibri.ttf"), ("calibrib.ttf"), ("calibriz.ttf"), ("calibrili.ttf"),
         ("OCRAEXT.TTF"), ("PERTIBD.TTF"), ("PERTILI.TTF"), ("segmdl2.ttf"),
         ("verdana.ttf"), ("TEMPSITC.TTF"), ("COPRGTL.TTF"), ("Inkfree.ttf"), 
         ("STXINWEI.TTF"), ("FZSTK.TTF"), ("Sitka.ttc"), ("SitkaZ.ttc"),
         ("PAPYRUS.TTF"), ("HTOWERT.TTF"), ("AGENCYB.TTF"), ("AGENCYR.TTF"),
         ("kaiu.ttf"), ("consola.ttf"), ("REFSAN.TTF")]


'''
Make fonts object
'''
Fonts = []
for f in FontNames:
    Fonts.append(ImageFont.truetype("C:/Windows/Fonts/" + f, 28))
    Fonts.append(ImageFont.truetype("C:/Windows/Fonts/" + f, 24))
  #  Fonts.append(ImageFont.ImageFont.truetype("C:/Windows/Fonts/" + f, 32))
NUM_FONTS = len(Fonts)

def relu(x):
    if x > 0:
        return x 
    else:
        return 0

def generate_vc(text : str, length = 4, image_width = VC_WIDTH, image_height = VC_HEIGHT):
    vc = Image.new('RGB', (image_width, image_height), '#FFFFFF')
    W = int(VC_WIDTH / length)
    tid = 0

    for t in text:
        font : ImageFont.FreeTypeFont = Fonts[random.randint(0, NUM_FONTS-1)]   #随机选一个字体
        tw, th = font.getsize(t)
        
        temp = Image.new('RGB', (tw + 6, th + 6), '#FFFFFF')
        draw2 = ImageDraw.Draw(temp)
        draw2.ink = random.randint(0,255) + random.randint(0, 255) * 256 + random.randint(0, 255) * 256
        draw2.text((0, 0), t, font=font)
        temp.rotate(random.randint(-5, 5))  #字体随机倾斜一点
        
        basex = tid * W + random.randint(0, relu(W-temp.width))
        basey = 6 + random.randint(0, relu((image_height-temp.height) - 6))
        vc.paste(temp, (basex, basey))
        temp.close()
        
        tid += 1

    #随机加入椒盐噪声
    draw = ImageDraw.Draw(vc)
    num_points = random.randint(10, 42)
    for _ in range(num_points):
        x = random.randint(0, image_width)
        y = random.randint(0, image_height)
        radiusx = random.randint(1, 1)
        radiusy = random.randint(1, 1)
        draw.rectangle((x, y, x+radiusx, y+radiusy), fill=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)))

    #随机加入线条干扰
    num_lines = random.randint(1, 6)
    for _ in range(num_lines):
        sx = random.randint(0, image_width)
        sy = random.randint(0, image_height)
        ang = random.randint(0, 360)
        len = random.randint(4, 30)
        dx = sx + int(len * math.cos(ang))
        dy = sy + int(len * math.sin(ang))
        draw.line([(sx, sy), (dx, dy)], fill=(random.randint(0, 255),random.randint(0, 255),random.randint(0, 255)), width=random.randint(1,2))

    return vc 

def random_seq(seqlen):
    return ''.join([ str(random.randint(0, 9)) for _ in range(seqlen)])

OUTPUT_DIR = "./validation_code_2/"

if __name__ == "__main__":
    #img = generate_vc("3CW9")
    #img.show()
    NUM_PICS = 5000

    keep_prob = 1.0  #输出原图文件的概率，可以命令行参数调低避免输出太多图片文件看着难受
    if len(sys.argv) > 1:
        keep_prob = float(sys.argv[1])
    if len(sys.argv) > 2:
        OUTPUT_DIR = "./{}/".format(sys.argv[2])
    
    imgs = np.ndarray(shape=(NUM_PICS, VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS))
    labels = []
    for Xid in range(NUM_PICS):
        text = ''.join(random.sample(CLASS_TO_NAME, k=4))
        labels.append(text)
        img = generate_vc(''.join(text))
        if np.random.rand() <= keep_prob:
            img.save("./{}/{}_{}.jpg".format(OUTPUT_DIR, random_seq(8), text), "jpeg")
        #gray = img.convert('L')
        imgs[Xid, :, :, :] = np.array(img).reshape(VC_HEIGHT, VC_WIDTH, VC_OCR_CHINNELS).astype('float32') / 255.0
        img.close()
        #gray.close()   

    np.save("./{}/packed_dataX".format(OUTPUT_DIR), imgs)
    with open("./{}/packed_dataY_text_label.pickle".format(OUTPUT_DIR), "wb") as f:
        pickle.dump(labels, f)
    
        
        
