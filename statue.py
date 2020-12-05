import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras.preprocessing.image import img_to_array
import face_recognition
import operator
import genderDetector
import functions

class SimilarityChecker():
    def __init__(self,imagePath):
        self.personGender = None
        self.beared = None
        self.imagePath = imagePath
        self.image = face_recognition.load_image_file(self.imagePath)
        self.image_encoding = face_recognition.face_encodings(self.image)[0]
        self.known_encodings = [self.image_encoding]
        self.dataset = pd.read_csv("static/dataset/csv/faces.csv")
        self.distances = dict()
        self.new_distances = None
        self.main()

    def findGender(self):  
        ## Finds person's gender 
        ## For filtering statue images
        modelGender = genderDetector.loadModel()
        img_224 = functions.preprocess_face(img =self.imagePath, target_size = (224, 224), grayscale = False,enforce_detection=False)
        genderPred = modelGender.predict(img_224)[0,:]
        if np.argmax(genderPred) == 0:
            self.personGender = "Woman"
        elif np.argmax(genderPred) == 1:
            self.personGender = "Man"


    def isBeard(self):
        image = cv2.imread(self.imagePath)
        image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        image_gray = cv2.cvtColor(image_gray,cv2.COLOR_GRAY2BGR)
        faces = face_cascade.detectMultiScale(image_gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
            roi_gray = image_gray[y:y+h, x:x+w]    
            roi_gray = cv2.resize(roi_gray,(64,64))
            roi_beard = roi_gray[35:90,7:55]
            roi_beard = cv2.resize(roi_beard,(28,28))
            roi_beard_array = img_to_array(roi_beard)
            roi_beard_array = roi_beard_array/255
            roi_beard_array = np.expand_dims(roi_beard_array,0)
            prediction = model.predict(roi_beard_array)
            if prediction[0][0]<0.5:
                self.beared = True
            else:
                self.beared = False
     

    def filterDataset(self):
        ## Filtering statue dataset by person's gender for reducing time and space comlexity
        ## With this filter, number of similarity decreasing
        if self.personGender == 'Man' and self.beared == True:
            self.dataset = self.dataset[self.dataset.gender == self.personGender]
            self.dataset = self.dataset[self.dataset.isBeard == str(self.beared)]
        else:
            self.dataset = self.dataset[self.dataset.gender == self.personGender]

    def calculateSimilarities(self):
        ## Calculating distances with dlib. User's photo vs. Statues
        for i in self.dataset.filename:
          try:
                i = f"static/{i}"
                image_to_test = face_recognition.load_image_file(i)
                image_to_test_encoding = face_recognition.face_encodings(image_to_test)[0]
                face_distances = face_recognition.face_distance(self.known_encodings,image_to_test_encoding)
                self.distances[i] = face_distances
          except:
            pass
    def sortDistances(self):
        ## Sorting distances for finding best match.
        new_dict = {}
        for i in self.distances:
          new_dict[i] = float(self.distances[i]) 
        self.distances = new_dict
        self.new_distances = sorted(self.distances.items(), key=operator.itemgetter(1))


    def plotOutput(self):
        ## Plotting output
        image = cv2.imread(self.new_distances[0][0]) 
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = cv2.resize(image,(500,500))
        image2 = cv2.imread(self.imagePath)
        image2 = cv2.cvtColor(image2,cv2.COLOR_BGR2RGB) 
        image2 = cv2.resize(image2,(500,500))
        f = plt.figure(figsize=(15,15))
        f.add_subplot(1,2, 1)
        plt.imshow(image2)
        plt.axis("off")
        f.add_subplot(1,2, 2)
        plt.imshow(image)
        plt.show(block=True)
        plt.axis("off")
        f.savefig("static/uploads/Result.png")
        plt.close("all")

    def main(self):
        print("Let's Roll")
        self.findGender()
        self.filterDataset()
        self.calculateSimilarities()
        self.sortDistances()
        print("Done !!!")
        self.plotOutput()

