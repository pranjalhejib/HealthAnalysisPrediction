import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from keras.models import load_model

model = load_model('NN.h5')

def get(file_path):
        result = 'Healthy Lungs'

        img = cv2.imread(file_path)

        height, width = img.shape[:2]

        img = cv2.resize(img, (100,100))

        # predict!
        roi_X = np.expand_dims(img, axis=0)
        predictions = model.predict(roi_X)

        #print(np.argmax(predictions[0]))
        result_index = np.argmax(predictions[0])
##        print (classes[result_index]) 
##        print(acc)
        per = 0
        if(result_index == 1):
                result = 'Lungs Cancer Detected'

                img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                (thresh, blackAndWhiteImage) = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
                count = cv2.countNonZero(blackAndWhiteImage)
                per =((count * 100)/ (height * width))
                per =  "%.2f" % round(per,2)
                plt.imshow(gray)
        return result,per

path='static/images/cancer/1001.png'
result,per = get(path)
img = cv2.imread(path)
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
plt.imshow(img)
print(result)
print(per)