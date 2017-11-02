**Behavioral Cloning** 

The idea of behavioral cloning is to learn by imitation. In this project, we train a Neural Network with images from cameras in a car and their associated recorded steering angles. Once the network is trained it can be used to predict steering angles based on new images received from the simulator and use it to guide the car.

The car should be able to drive the track without any mistakes. More specifically, from the Udacity Self-Driving Car Nanodegree project specification: "No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)."
---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road

**Data collection**

The training data is recorded using Udacity's simulator.
![image1]:(./misc/simulator.JPG "Simulator")

The simulator on record mode in training creates a folder "IMG" which contains the image data from the center, left and right. It also creates a .csv file that contains the path to these images along with its corresponding steering angle, throttle, brake and speed.
[image2]: ./misc/drivinglog.JPG "CSVfile"

A total of 17339 datasets were collected using the Simulator.

**Data Augmentation**

The following techniques were used to augment the dataset. 
1. Used the left and right images from the dataset to train the recovery driving scenario. The dataset recorded from the simulator would have steering angle for the left and right images as to the one corresponding the center image. So, for every left image, the new steering angle is adjusted by +0.2 and for every right image the new steering angle is adjusted ny -0.2.

[image3]: ./misc/center.JPG "Center Image" 
[image4]: ./misc/left.JPG "Left Image" 
[image5]: ./misc/right.JPG "Right Image" 


2. The center, left and right images were flipped horizontally to further augment the data. The steering angle for the flipped images were set to be the negative of the non flipped images.

[image6]: ./misc/flip.JPG "Flipped Image" 

The data augmentation resulted in creating a total of 104034 datasets.

**Data Preprocessing**

The preprocessing step consistes of resizing the images from 160x320 to 100x200 followed bu converting the BGR image that tensor flow reads in to RGB image. In addition to this step, the images are normalized by dividing the image data by 255.0 and subtracting 0.5 from it. The mean normalization step (to avoid saturation and make gradients work better) and the cropping of the image to remove the sky and hood of the car from the image is part of the network model. (Layer 1 and 2)

[image7]: ./misc/beforecrop.JPG 
[image8]: ./misc/aftercrop.JPG 

**Neural Network Model : NVIDIA**

The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project.

The netwok receives input images of shape (100,200,3) and falls through the lambda layer which does the mean normalization. The lambda layer is followed by a cropping layer that crops out the 100x200 image to (66, 200, 3). The cropping layer removes the sky and the hood of the car from the input image. The cropping layer is followed by 5 convolutional layers with RELU activation units, followed by 4 fully connected layers. The convolution layers 1, 2 and 3 has (5, 5) filters and a stride of 2, whereas the convolution layers 4 nd 5 has (3, 3) filters and stride of 1.

The neural network was implemented using the Keras API on top of TensorFlow. The below figure shows the model architecture described above.

[image9]: ./misc/modelarch.JPG "Model Architecture"

**Model training parameters**
The neural network described above was trained using the following parameters:

Number of epochs: 5
Batch size: 32
Optimizer: Adam, learning rate: default 1e-3
Validation split: 20% of the dataset.
Generator: I used a data generator for Keras. The purpose is to avoid having all the image data in memory at any point in time, since for large datasets it could consume all system memory resources.
Once the network is trained the model is saved to disk (model.json and model.h5) so that drive.py can load it and use it for predicting steering angles.

**Attempts to reduce overfitting in the model**
The validation loss was found to decrease throughout the 5 epochs and no oscillations were observed(signs of overfitting). So no dropout layers are implemented in this project. Limiting the epochs to 5 helped the model from overfitting.

**Training and Validation Set**

I drove around the simulator track (center lane driving) for 8 laps in clockwise direction and 8 laps in anti clockwise direction. I also recorded recovery scenarios like driving the car from the side of the track back to the center. The dataset is shuffled and split into training and validation set in 8:2 ratio.

**Visualizing the training loss and validation loss**

I used the history object of model.generator() to output a plot showing how the training loss and validation loss changed with the epochs.

**Using the Neural Network to drive the car**
The python code that predicts the steering angles and drive the car is implemented in drive.py. The template for this file was provided by Udacity. It loads the trained neural network model from disk (model.h5) and uses the images provided by the simulator (which is connected to drive.py using a socket) to predict the steering angle. The implementation in drive.py also has to provide a throttle for the simulator, otherwise the car would not move. The speed was set to a constant 9.0.

I added the preprocessing step - resizing the 160x320 image to 100x200 image in the drive.py python code, so that the model is tested on the same image dimensions as the training set.

**Conclusion**
This project shows how to use a deep learning neural network to teach a car how to drive itself. After training the network using data from one track 1, the car was able to drive itself meeting the UDACITY specification - "No tire may leave the drivable portion of the track surface. The car may not pop up onto ledges or roll over any surfaces that would otherwise be considered unsafe (if humans were in the vehicle)."

Future updates could include training the model on the second track and using the data set from track1 along with track2 to generalize the model and use the network to drive track2 without any mistakes. A variable throttle function can also be implemented to set the speed of the car while drving autonomously.
 
---
**Quick Run**
The project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* video.py used to create video of the final output

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```
To train the model
```sh
python model.py
```
This will generate model.h5 which can be used with drive.py.
