from tensorflow import keras
from keras import layers
from keras.preprocessing.image import ImageDataGenerator

numClasses = 12

#Define the paths to the training and testing directories
#Normal images
trainDirectory= "../Prova.v1i.folder/train"
testDirectory= "../Prova.v1i.folder/test"

#Gabor images
#train_dir= "../ProvaGabor/train"
#test_dir= "../ProvaGabor/test"


#Set the image size and batch size
k=128
imageSize=(k, k)
batchSize=6

#To load the dataset we've followed 
# https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator#flow_from_directory


#Create the ImageDataGenerator for training data with data augmentation
trainDatagenerator=ImageDataGenerator(
    rescale=1.0/255.0, #Rescaling to this value because it's the maximum value of pixels
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
    vertical_flip=False
)

#Load the training data from the directory with data augmentation
trainSet=trainDatagenerator.flow_from_directory(
    trainDirectory,
    target_size=imageSize,
    batch_size=batchSize,
    class_mode='categorical'
)

#Create the ImageDataGenerator for testing data without data augmentation
testDatagenerator=ImageDataGenerator(rescale=1.0/255.0)

#Load the testing data from the directory without data augmentation
testSet=testDatagenerator.flow_from_directory(
    testDirectory,
    target_size=imageSize,
    batch_size=batchSize,
    class_mode='categorical'
)


#Define the model architecture. We tried different pooling
model = keras.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(k, k, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    #layers.MaxPooling2D((2,2)),
    layers.AveragePooling2D((2,2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(numClasses, activation='softmax')
])

#Setting Metrics
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#Train the model with augmented data, we tried more epochs
model.fit(trainSet, epochs=100)

#Evaluate the model
testLoss, testAccuracy = model.evaluate(testSet)
print("Test Loss:", testLoss)
print("Test Accuracy:", testAccuracy)

#Save the model
model.save("multiCNN.keras")