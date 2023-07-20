import numpy as np
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
import cv2
import sys

# Caricamento del modello
model = keras.models.load_model("multiCNN.keras")

#Threshold
threshold = 0.05
#Image size for input
k = 128

#Code to load class name/Labels, not elegant at all since it loads the test set every time.
#Plus I can't seem to turn off the option to print the found images and classes.
#Same code is as in the training of the model (multiCNN.py)
imageSize = (k, k)
batchSize = 6
testDatagen = ImageDataGenerator(rescale=1.0/255.0)
testSet = testDatagen.flow_from_directory(
    "../Prova.v1i.folder/test",
    target_size=imageSize,
    batch_size=batchSize,
    class_mode='categorical',
    shuffle=False
)

#labels
classIndices = testSet.class_indices
labels = list(classIndices.keys())

#Counting how many bounding boxes we find
#In foodImage
nBB=0
try:
    with open("../comparison/bounding_boxes/food_image_bounding_box.txt", 'r') as file:
        for i in file:
            nBB+=1
except FileNotFoundError:
    print("Error in loading the n. of Bounding boxes of FoodImage")
    sys.exit()

#In leftover
nBBL=0
try:
    with open("../comparison/bounding_boxes/leftover_bounding_box.txt", 'r') as file:
        for i in file:
            nBBL+=1
except FileNotFoundError:
    print("Error in loading the n. of Bounding boxes of Leftover")
    sys.exit()
#Creating the text file for the predictions
open("../comparison/bounding_boxes/food_image_bounding_box_classification.txt", 'w')
open("../comparison/bounding_boxes/leftover_bounding_box_classification.txt", 'w')

#Predictions
#For BB in food image
print("==============")
print("Printing the classification of the Bounding Boxes found in the Food Image:")
for ii in range(0, nBB):
    try:
        img = cv2.imread("../comparison/masks/food_image_masks/maskBB_{}.jpg".format(ii))
        
        #Image Preprocessing
        preprocessedImg = cv2.resize(img, (k, k))
        preprocessedImg = preprocessedImg.astype('float32') / 255.0
        preprocessedImg = np.expand_dims(preprocessedImg, axis=0)  # Espandi le dimensioni dell'immagine
        
        #Prediction
        prediction = model.predict(preprocessedImg)
        
        predict = []
        for i, prob in enumerate(prediction[0]):
            if prob > threshold:
                predict.append(labels[i])
        
        print("Prediction:", predict)

        #Save the results on a file
        try:
            with open("../comparison/bounding_boxes/food_image_bounding_box_classification.txt", 'a') as file:
                text="["
                file.write(text)
                for cl in predict:
                    file.write(" "+cl+", ")
                file.write("]\n")
        except Exception as e:
            print(f"Error in writing the classes on a file: {e}")
            sys.exit()

    except:
        continue

#For BB in leftover image
for ii in range(0, nBBL):
    print("==============")
    print("Printing the classification of the Bounding Boxes found in the Leftover Image:")
    try:
        img = cv2.imread("../comparison/masks/leftover_masks/maskBB_{}.jpg".format(ii))
        
        #Image Preprocessing
        preprocessedImg = cv2.resize(img, (k, k))
        preprocessedImg = preprocessedImg.astype('float32') / 255.0
        preprocessedImg = np.expand_dims(preprocessedImg, axis=0)  # Espandi le dimensioni dell'immagine
        
        #Prediction
        prediction = model.predict(preprocessedImg)
        
        predict = []
        for i, prob in enumerate(prediction[0]):
            if prob > threshold:
                predict.append(labels[i])
        
        print("Prediction:", predict)

        #Save the results on a file
        try:
            with open("../comparison/bounding_boxes/leftover_bounding_box_classification.txt", 'a') as file:
                text="["
                file.write(text)
                for cl in predict:
                    file.write(" "+cl+" ")
                file.write("]\n")
        except Exception as e:
            print(f"Error in writing the classes on a file 2: {e}")
            sys.exit()
    except:
        continue