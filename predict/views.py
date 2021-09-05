from django.shortcuts import render
from tensorflow.keras.models import load_model
from .models import Files, Class_index
import cv2, imutils, joblib
import numpy as np

def removal(image):
    img = cv2.imread(image)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    hh, ww = thresh.shape
    thresh[hh-3:hh, 0:ww] = 0
    white = np.where(thresh==255)
    xmin, ymin, xmax, ymax = np.min(white[1]), np.min(white[0]), np.max(white[1]), np.max(white[0])
    crop = img[ymin:ymax+3, xmin:xmax]
    return crop

def preprocessing(image):
    img = cv2.resize(image,(500,500))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = np.array(255*(img/255)**0.5,dtype='uint8')
    img = cv2.GaussianBlur(img,(3,3),0)
    hist = cv2.calcHist([img],[0],None,[256],[0,256]) 
    if imutils.is_cv2():
        hist = cv2.normalize(hist)
    else:
        cv2.normalize(hist, hist)
    hist = hist.flatten()
    hist = np.reshape(hist,(1,256))
    return hist

Model_healthy = load_model('./model/ANN_Healthy_2')
Model_COVID = joblib.load('./model/SVM_COVID.model')

def filterflow(hist):
    Hpred = Model_healthy.predict(hist)
    Hpred = np.argmax(Hpred,axis=1)

    if Hpred == 1:
        result = 'Non-infected'

    elif Hpred == 0:
        Cpred = Model_COVID.predict(hist)

        if Cpred == 0:
            result = 'Typical COVID-19 infected'
        else:
            result = 'Pneumonia infected'
    
    return result

def index(request):
    return render(request, 'index.html')

def prediction(request):
    if request.method == 'POST':
        images = request.FILES.getlist('images')

        Allobj = []

        for i in images:
            new_file = Files(images=i)
            new_file.save()

            obj = Class_index()
            obj.path = new_file.images.url
            obj.title = i.name

            img = '.'+new_file.images.url
            img = removal(img)
            hist = preprocessing(img)
            obj.result = filterflow(hist)

            Allobj.append(obj)
 
    context = {'Allobj':Allobj}
    return render(request, 'index.html',context)