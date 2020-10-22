# -*- coding: utf-8 -*-
"""
Created on Sat Jul 11 14:44:24 2020

@author: Shalin
"""

import numpy as np
import cv2
import pickle
 
########### PARAMETERS ##############
width = 640
height = 480
threshold = 0.65 # MINIMUM PROBABILITY TO CLASSIFY
cameraNo = 0
#####################################
 
#### CREATE CAMERA OBJECT
cap = cv2.VideoCapture(cameraNo)
cap.set(3,width)
cap.set(4,height)
dict = {
            'item1': 1
}
flag = 0;
id = "no matching"
 
#### LOAD THE TRAINNED MODEL 
pickle_in = open("Maskmodel_trained.p","rb")
model = pickle.load(pickle_in)
 

#### PREPORCESSING FUNCTION
def preProcessing(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.equalizeHist(img)
    img = img/255
    return img

 
while True:
    success, imgOriginal = cap.read()
    img = np.asarray(imgOriginal)
    img = cv2.resize(img,(32,32))
    img = preProcessing(img)
    cv2.imshow("Processsed Image",img)
    img = img.reshape(1,32,32,1)
    #### PREDICT
    id = int(model.predict_classes(img))
    #print(classIndex)
    predictions = model.predict(img)
    #print(predictions)
    probVal= np.amax(predictions)
    print(id,probVal)
    
    
    if probVal> threshold:
        if(id==0):
            id = 'Mask'
            if ((str(id)) not in dict):
                #filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);
            cv2.putText(imgOriginal,str(id) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1.5,(0,255,0),2)
                
        elif(id==1):
            id = 'No_Mask'
            if ((str(id)) not in dict):
                #filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);
            cv2.putText(imgOriginal,str(id) + "   "+str(probVal),
                    (50,50),cv2.FONT_HERSHEY_COMPLEX,
                    1.5,(0,0,255),2)
            
        else:
            id = 'No face'
            flag=flag+1
            if ((str(id)) not in dict):
                #filename =xlwrite.output('attendance', 'class1', 4, id, 'yes');
                dict[str(id)] = str(id);
                cv2.putText(imgOriginal,str(id) + "   "+str(probVal),
                        (50,50),cv2.FONT_HERSHEY_COMPLEX,
                        1,(0,0,255),2)
                
                break
     
        cv2.imshow("Original Image",imgOriginal)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cap.release()
cv2.destroyAllWindows()
