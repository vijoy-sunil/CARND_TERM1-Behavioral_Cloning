import csv
import cv2
import random
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

STEER_CORRECTION = 0.2
BATCH_SIZE = 32
AUGMENTATION_FACTOR = 6
EPOCHS = 5

#Load .csv file
samples = []
with open('./my_data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

#histogram view
#range = (-1, 1)
#bins = 100 
#plt.hist(y_train, bins, range, color = 'red', histtype = 'bar', rwidth = 0.8)
#plt.ylabel('No. of examples')
#plt.xlabel('Steering angle')
#plt.show()

train_samples, validation_samples = train_test_split(samples, test_size = 0.2)                                    
TRAIN_SAMPLES_LEN = AUGMENTATION_FACTOR * len(train_samples)
VALIDATION_SAMPLES_LEN = AUGMENTATION_FACTOR * len(validation_samples)

def preprocess(image):
    output_shape = (200, 100)
    output = cv2.resize(image, output_shape, interpolation = cv2.INTER_AREA)
    rgb_image = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)   
    return rgb_image

#Generator function
def generator(samples, batch_size = BATCH_SIZE):
    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset: offset + batch_size]
            images, angles = [], []

            for batch_sample in batch_samples:
                for i in range(3):
                    source_path = batch_sample[i]                       #image - center|left|right
                    filename = source_path.split('/')[-1]
                    current_path = './my_data/IMG/' + filename
                    image = cv2.imread(current_path)
                               
                    preprocess_image = preprocess(image)            
                    images.append(preprocess_image)
                    images.append(cv2.flip(preprocess_image,1))         #augment data by flipping
            
                angle = float(line[3])                                  #steering angle

                angle_center = angle
                angles.append(angle_center)            
                angles.append(angle_center * -1.0)

                angle_left = angle + STEER_CORRECTION
                angles.append(angle_left)            
                angles.append(angle_left * -1.0)

                angle_right = angle - STEER_CORRECTION
                angles.append(angle_right)            
                angles.append(angle_right * -1.0)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size = BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size = BATCH_SIZE)

#network architecture
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Activation, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D

model = Sequential()
#Normalization layer
model.add(Lambda (lambda x: x/255.0 - 0.5, input_shape = (100,200,3)))

#Cropping
model.add(Cropping2D (cropping = ((24,10),(0,0))))

model.add(Convolution2D(24, 5, 5, subsample = (2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(36, 5, 5, subsample = (2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(48, 5, 5, subsample = (2, 2)))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Convolution2D(64, 3, 3))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))


model.compile(loss = 'mse', optimizer = 'adam')
history_object = model.fit_generator(train_generator, \
                                    samples_per_epoch = TRAIN_SAMPLES_LEN,\
                                    validation_data = validation_generator,\
                                    nb_val_samples = VALIDATION_SAMPLES_LEN,\
                                    verbose = 1, nb_epoch = EPOCHS)

print('Model saved')
model.save('model.h5')

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()

                                 
