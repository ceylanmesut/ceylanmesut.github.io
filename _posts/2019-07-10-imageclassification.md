---
title: "Intel Image Data Classification Project"
date: 2019-09-13
tags: Computer Vision, Machine Learning, Convolutional Neural Network]
header:
 image: "/images/house.png"
excerpt: "Image Classification Project on Landscape Images."

toc: true
toc_label: " On This Page"
toc_icon: "file-alt"
toc_sticky: true
---

## Introduction
Main idea of this project is to successfully classify intel landscape image dataset. This dataset consists of 6 different landscapes namely; **buildings, streets, glaciers, forests, deserts and XX** and I'm going to use **Convolutional Neural Networks (ConvNets)** machine learning method to classify these images **as fast as and as accurate as possible.**

Convolutional Neural Network is **special type of Artificial Neural Network (ANN)** structure.
What separates Convolutional Neural Networks from Artificial Neural Networks is state of art structure of **ConvNets that is specifically created for image classification and related tasks.** Unlike ANN's fully connected network structure, **Cluster of Convolutional Layers is the core of ConvNets.** and it is the main engine to squeeze the images into processable size and structure. Not surprisingly, this unique structure boosts computational capability of ConvNets during image classification tasks when it compared to ANN.


* **Dataset**: Intel image dataset includes 6 different landscape images with 150x150 size.


* **Inspiration**: Accurately classify as much as image possible with robust machine learning.


* **Problem Definition**: Building Convolutional Neural Network model to obtain high accuracy.


* **Link**: https://www.kaggle.com/puneet6060/intel-image-classification


## Approach
* **0.Explanatory Data Analysis**: Understanding the dataset and check class imbalance.


* **Convolutional Neural Network**: Creating **ConvNets model** for the problem.


* **Hyperparameter Tuning**: Optimizing **hyperparameters** of the ConvNets model to achieve better results.

## Models
* **ConvNets**:  **Variants of ConvNets** models.


### EDA

Let's load necessary packages.

``` python
# Import necessary libraries and packages
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2  
import keras
import tensorflow.keras.layers as Layers
import tensorflow.keras.models as Models
import tensorflow.keras.optimizers as Optimizer
from keras.regularizers import l1
from keras.layers.normalization import BatchNormalization
import tensorflow.keras.metrics as Metrics
from keras.layers import Dropout
from sklearn.utils import shuffle
from sklearn.model_selection import cross_val_score
import seaborn as sn
import timeit
import os
from keras.optimizers import Adam
from random import randint

# Defining paths to tranning and test images.
TRAIN_PATH = "../input/seg_train/seg_train/"
TEST_PATH = "../input/seg_test/seg_test/"
```

So far so good. Next, let's **discover image categories** by their percentage distributions.
``` python
# Exploratory analysis
def explore_categories(path):
    """This function explores data folders and counts number of landscape category by category."""

    # Counting each iamge category
    for category in os.listdir(path):
        if(category == "buildings"):
            no_buildings = len(os.listdir(path + "/" + "buildings"))
        elif(category == "forest"):
            no_forest = len(os.listdir(path + "/" + "forest"))
        elif(category == "glacier"):
            no_glacier = len(os.listdir(path + "/" + "glacier"))  
        elif(category == "mountain"):
            no_mountain = len(os.listdir(path + "/" + "mountain"))
        elif(category == "sea"):
            no_sea = len(os.listdir(path + "/" + "sea"))   
        elif(category == "street"):
            no_street = len(os.listdir(path + "/" + "street"))

    # Summing all images.        
    total_images = no_buildings + no_forest + no_glacier + no_mountain + no_sea + no_street

    # Pie chart, where the slices will be ordered and plotted counter-clockwise:
    labels = 'Buildings', 'Forest', 'Glacier', 'Mountain', 'Sea', 'Street'
    percentages = [no_buildings/total_images, no_forest/total_images, no_glacier/total_images, no_mountain/total_images, no_sea/total_images, no_street/total_images]

    if(path == TEST_PATH):
        pie_chart_generate(percentages, labels, "Test Data")
    elif(path == TRAIN_PATH):
        pie_chart_generate(percentages, labels, "Training Data")
    return total_images


def pie_chart_generate(percentages, labels, title):
  """This function generates pie charts of given class labels."""
    # Defining color map for pie chart.
    cmap = plt.get_cmap("tab20c")
    outer_colors = inner_colors = cmap(np.array([1, 2, 5, 6, 9, 10]))

    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.pie(percentages, labels=labels, autopct='%1.1f%%',
        shadow=False, startangle=90, colors=outer_colors)
    ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()


# Training data pie chart
number_training_images = explore_categories(TRAIN_PATH)
# Testing data pie chart
number_testing_images = explore_categories(TEST_PATH)

# Pie chart of the ratio of training and testing data
training_testing_ratio = [number_training_images/(number_training_images + number_testing_images), number_testing_images/(number_training_images + number_testing_images)]
pie_chart_generate(training_testing_ratio, ['Training Data', 'Test data'], 'Training-Test Ratio')

print("Number of training images: " + str(number_training_images))
print("Number of testing images: " + str(number_testing_images))
print("Number of images for prediction: " + str(len(os.listdir("../input/seg_pred/seg_pred/"))))
```



<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/pie1.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/pie2.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/pie3.png" alt="">

Well, clearly there is **no class imbalance on both training and test images** so it is good news. Also we can see that we have **high amount of training images** and low amount of test images so that I need to be careful with **overfitting of the model.**

Next, let's load the data from paths that I defined and shuffle data.

``` python
#Preprocess data
def pre_process(path, image_size=100):
    """This function loads, resizes, standardizes and shuffles all images."""
    data = []
    labels = []
    for category in os.listdir(path):
        if(category == "buildings"):
            label = 0
        elif(category == "forest"):
            label = 1
        elif(category == "glacier"):
            label = 2  
        elif(category == "mountain"):
            label = 3  
        elif(category == "sea"):
            label = 4   
        elif(category == "street"):
            label = 5

        training_subfolder_path = path + "/" + category

        for file in os.listdir(training_subfolder_path):
            image_path = training_subfolder_path + "/" + file
            image = cv2.imread(image_path)

            #Resize all images so they all have the same size
            image = cv2.resize(image,(image_size, image_size))
            image = np.array(image)

            #Standardize data by dividing by 255
            image = image.astype('float32')/255.0
            data.append(image)
            labels.append(label)

    #Shuffle data
    data, labels = shuffle(data, labels)
    data = np.array(data)
    labels = np.array(labels)
    return data, labels
```

Let's load the data.

```python
# Loading data
train_data, labels = pre_process(TRAIN_PATH, image_size=100)
```

Let's assign class labels to each image and plot some of them to check the assign labels.

```python
def get_classlabel(class_code):
  """This function assign class label text on every image according  to their type."""
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}  
    return labels[class_code]
# Plotting images with class labels.
f,ax = plt.subplots(3,3)
f.subplots_adjust(0,0,3,3)
for i in range(0,3,1):
    for j in range(0,3,1):
        rnd_number = randint(0,len(train_data))
        ax[i,j].imshow(train_data[rnd_number])
        ax[i,j].set_title(get_classlabel(labels[rnd_number]))
        ax[i,j].axis('off')
```



![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cl_images.png){: .align-center}

Alright, let's start to build our first Convolutional Neural Network. Before constructing the model, I would like to introduce core elements of ConvNets structure.
* **Convolutional Layer:** **Fundamental component** of ConvNets. These layers are responsible for **filtering given input image and capturing certain features** of the image via applying filter operation. Essentially, Conv Layers' role is filtering useful information from given input image.

GIFFF
* **Pooling Layers :** These layers are responsible for **reducing the number of parameters** of feature map that we obtained after convolutional layer. They function as **iterating specific kernel over feature map** to **apply function** on the map. Although there are different **types** such as **Max, Average and Sum pooling,** I used **Max Pooling** in which kernel iterates over rectified feature map and **takes largest elements of zone** that kernel applies its function.

GIFF




* **Activation Functions:** They introduces **non-linearity** into neural network structure. Their role is to **transform input signal of a node into output signal.** Introducing non-linearity into NN structure is **crucial to be able to induce learning of complex non-linear relation of input and output.** Most common activation functions are **Sigmoid** (Logistic), **Tanh** (Hyperbolic Tangent) and **ReLu** (Rectified Linear Units).

![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/acv_fun2.png){: .align-center}

* **Dropout:** Simply this **layer dropouts some of the nodes (units)** within neural network structure with **certain probability** while **forward and backward propagation.** Dropout layer is essentially included within model to **avoid overfitting** because **deeply connected and inter-dependent nodes naturally cause overfitting** through each training state.

![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/dropout.png){: .align-center}

Dropout image: Srivastava, Nitish, et al. ”Dropout: a simple way to prevent neural networks from overfitting”, JMLR 2014

* **Adam Optimizer:** Adam optimizer is one of the **most popular optimization method** being used training deep neural networks. Fundamentally, it is combination of **RMSprop and Stochastic Gradient Descend  with momentum.** It is **adaptive learning rate method** in which **individual learning rates** are computed for different parameters. It leverages first and second moments of gradient computations and use them to adapt the learning rate.


According to discussed structure ConvNets, I create below neural network to train my model and conduct predictions.

``` python
# Constructing Convolutional Neural Network Model
def cnn_model():
    """First Convolutional Nueral Network Model"""
    model = Models.Sequential()

    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))

    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))

    model.add(Layers.Flatten())
    model.add(Layers.Dense(256,activation='relu'))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(256,activation='relu'))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(6,activation='softmax'))
    model.compile(optimizer=Optimizer.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=False),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])
    return model
```

So far so good. I constructed my Convolutional Neural Network structure with Adam optimizer and proper learning rate. Next, I define model fit function to
fit the neural network and make prediction. The function also plots accuracy and loss outcomes along with confusion matrix.


```python
# Let's define model fit function.
def model_fit(my_model, number_epochs, batch_size):
    """This function accepts neural network structure, number of epochs and bathc size as function parameters and train the neural network."""
    start_time = timeit.default_timer()
    # Fit model
    model= my_model
    trained = model.fit(train_data,labels,epochs=number_epochs,validation_split=0.25,batch_size=batch_size)
    elapsed = timeit.default_timer()
    print('Runtime:', elapsed)

    # Plotting accuracy and validation accuracy.
    plt.plot(trained.history['acc'])
    plt.plot(trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plotting loss and validation loss.
    plt.plot(trained.history['loss'])
    plt.plot(trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Prediction on test set.
    test_images,test_labels = load_data('../input/seg_test/seg_test/')
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    model.evaluate(test_images,test_labels, verbose=1)

    # Plotting Confusion Matrix.
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis = 1)
    frame={'y':test_labels,'y_predicted':pred_labels}
    df = pd.DataFrame(frame, columns=['y','y_predicted'])
    confusion_matrix = pd.crosstab(df['y'], df['y_predicted'],rownames=['True Label'], colnames=['Predicted Label'], margins = False)
    sn.heatmap(confusion_matrix,annot=True,fmt="d",cmap="Blues",linecolor="blue", vmin=0,vmax=500)
    plt.title('Confusion Matrix', fontsize=16)
```

Now, let's run first prediction with defined neural network. I run my model for 15 epochs with 32 batch size.

```python
# First Prediction
model=cnn_model()
number_epochs=15
batch_size=32
model_fit(model, number_epochs,batch_size)
```

Let's take a look my model summary and parameters.


Model Summary
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
conv2d_40 (Conv2D)           (None, 98, 98, 128)       3584      
_________________________________________________________________
conv2d_41 (Conv2D)           (None, 96, 96, 128)       147584    
_________________________________________________________________
max_pooling2d_16 (MaxPooling (None, 32, 32, 128)       0         
_________________________________________________________________
conv2d_42 (Conv2D)           (None, 30, 30, 256)       295168    
_________________________________________________________________
conv2d_43 (Conv2D)           (None, 28, 28, 256)       590080    
_________________________________________________________________
max_pooling2d_17 (MaxPooling (None, 9, 9, 256)         0         
_________________________________________________________________
flatten_9 (Flatten)          (None, 20736)             0         
_________________________________________________________________
dense_26 (Dense)             (None, 256)               5308672   
_________________________________________________________________
dropout_17 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_27 (Dense)             (None, 256)               65792     
_________________________________________________________________
dropout_18 (Dropout)         (None, 256)               0         
_________________________________________________________________
dense_28 (Dense)             (None, 6)                 1542      
=================================================================
Total params: 6,412,422
Trainable params: 6,412,422
Non-trainable params: 0


At the moment, we have 10525 training image and 3509 validation image.

```python
Train on 10525 samples, validate on 3509 samples
Epoch 1/15
10525/10525 [==============================] - 17s 2ms/sample - loss: 1.2295 - acc: 0.4968 - val_loss: 0.9556 - val_acc: 0.6022
Epoch 2/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.9637 - acc: 0.6113 - val_loss: 0.8318 - val_acc: 0.6566
Epoch 3/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.8167 - acc: 0.6862 - val_loss: 0.7628 - val_acc: 0.7244
Epoch 4/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.7039 - acc: 0.7385 - val_loss: 0.5879 - val_acc: 0.7885
Epoch 5/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.6204 - acc: 0.7769 - val_loss: 0.5494 - val_acc: 0.8054
Epoch 6/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.5450 - acc: 0.8058 - val_loss: 0.5359 - val_acc: 0.8119
Epoch 7/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.4795 - acc: 0.8276 - val_loss: 0.5348 - val_acc: 0.8091
Epoch 8/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.4039 - acc: 0.8571 - val_loss: 0.5198 - val_acc: 0.8259
Epoch 9/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.3538 - acc: 0.8746 - val_loss: 0.5324 - val_acc: 0.8282
Epoch 10/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.3164 - acc: 0.8898 - val_loss: 0.5877 - val_acc: 0.8128
Epoch 11/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.2850 - acc: 0.9028 - val_loss: 0.7109 - val_acc: 0.7888
Epoch 12/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.2332 - acc: 0.9189 - val_loss: 0.6361 - val_acc: 0.8176
Epoch 13/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.1996 - acc: 0.9286 - val_loss: 0.8162 - val_acc: 0.7903
Epoch 14/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.1849 - acc: 0.9376 - val_loss: 0.7688 - val_acc: 0.8176
Epoch 15/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 0.1574 - acc: 0.9457 - val_loss: 0.7745 - val_acc: 0.8145
Runtime: 108304.077418699
```

Let's analyze model outcomes. **Clearly**, my model starts to **overfitting from 5th Epoch** as train and test lines **cross** each other and **builds separation** through following epochs. Therefore, it is easy to observe that model is **overfitting to training set** and it has poor performance on validation set.

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m1_acc.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m1_loss.png" alt="">

```python
3000/3000 [==============================] - 2s 508us/sample - loss: 0.7719 - acc: 0.8067
```
Overall, I obtain %80 accuracy from first prediction. As a baseline score, it is not bad but requires improvement.

![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cm_1.png){: .align-center}

From confusion matrix, one can observe that model **performs poorly** on recognizing **Building images** (Label 0) and mountain image (label 3). It misclassifies **mountains as glaciers** and **buildings as streets** or vice versa.


As next step, let's increase the batch size to boost batch of images that being trained in each step. I increase batch size from 32 to 128.

```python
# Second Prediction
model=cnn_model()
number_epochs=15
batch_size=128
model_fit(model, number_epochs,batch_size)
```

At this stage, I did not change structure of my model.

```python
Train on 10525 samples, validate on 3509 samples
Epoch 1/15
10525/10525 [==============================] - 15s 1ms/sample - loss: 1.3919 - acc: 0.4217 - val_loss: 1.0312 - val_acc: 0.5919
Epoch 2/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 1.0348 - acc: 0.5850 - val_loss: 0.8856 - val_acc: 0.6669
Epoch 3/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.9198 - acc: 0.6383 - val_loss: 0.7974 - val_acc: 0.6948
Epoch 4/15
10525/10525 [==============================] - 13s 1ms/sample - loss: 0.8045 - acc: 0.6896 - val_loss: 0.6988 - val_acc: 0.7378
Epoch 5/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.6838 - acc: 0.7540 - val_loss: 0.6444 - val_acc: 0.7729
Epoch 6/15
10525/10525 [==============================] - 13s 1ms/sample - loss: 0.6175 - acc: 0.7853 - val_loss: 0.5669 - val_acc: 0.7968
Epoch 7/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.5610 - acc: 0.8050 - val_loss: 0.5511 - val_acc: 0.8019
Epoch 8/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.4851 - acc: 0.8310 - val_loss: 0.5007 - val_acc: 0.8236
Epoch 9/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.4219 - acc: 0.8548 - val_loss: 0.4971 - val_acc: 0.8287
Epoch 10/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.3871 - acc: 0.8620 - val_loss: 0.4914 - val_acc: 0.8321
Epoch 11/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.3481 - acc: 0.8775 - val_loss: 0.5058 - val_acc: 0.8213
Epoch 12/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.2962 - acc: 0.8966 - val_loss: 0.4866 - val_acc: 0.8427
Epoch 13/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.2715 - acc: 0.9031 - val_loss: 0.5515 - val_acc: 0.8290
Epoch 14/15
10525/10525 [==============================] - 14s 1ms/sample - loss: 0.2365 - acc: 0.9158 - val_loss: 0.5132 - val_acc: 0.8521
Epoch 15/15
10525/10525 [==============================] - 13s 1ms/sample - loss: 0.2051 - acc: 0.9269 - val_loss: 0.5623 - val_acc: 0.8487
Runtime: 108832.818498101
```
Alright, **overfitting** problem is **still evident** fact from 7th Epoch.

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m2_acc.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m2_loss.png" alt="">


```python
3000/3000 [==============================] - 2s 504us/sample - loss: 0.5735 - acc: 0.8433
```
Yet, model manages to decrease **loss from 0.77 to 0.57** and to **increase accuracy almost %4.** This is great!. Now, the model correctly **classifies %84 of images.**

![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cm_2.png){: .align-center}

Reflection of model accuracy increase can be observed from confusion matrix as well. Number of correct classification of building (label 0) and mountain (label 3) increased.

As I am looking forward **to increase my model accuracy,** I start applying **Data Augmentation** to increase my training and validation data. Data Augmentation is a method to increase available dataset by altering image specification of existing image. **Alteration** may involve:
* Horizontal or vertical flip,
* Gamma adjustment,
* Rotation of image,
* Adding Gaussian noise,
* Cropping, zooming and stretching.

In my model, I only benefit from flipping images horizontally and vertically. I observed **decrease on accuracy** when I applied **gamma adjustment, zooming and sheering.**


```python
# Data Augmentation Section
import random
from scipy import ndarray
import skimage as sk
from skimage import transform
from skimage import util
from skimage.exposure import adjust_gamma

#Definning augmentation operations.
def horizontal_flip(image):
    """Flips the given image horizontally"""
    return image[:, ::-1]

def up_side_down(image):
    return np.rot90(image, 2)

# Defining augmentation methods.    
methods={'h_flip':horizontal_flip,'u_s_d':up_side_down}

data = []
labels = []

path = "../input/seg_train/seg_train/"
for category in os.listdir(path):
    if(category == "buildings"):
        label = 0
    elif(category == "forest"):
        label = 1
    elif(category == "glacier"):
        label = 2  
    elif(category == "mountain"):
        label = 3  
    elif(category == "sea"):
        label = 4   
    elif(category == "street"):
        label = 5

    training_subfolder_path = path + "/" + category        
    for file in os.listdir(training_subfolder_path):
        image_path = training_subfolder_path + "/" + file
        image = cv2.imread(image_path)

        #Resize all images so they all have the same size
        image = cv2.resize(image,(100,100))
        image = np.array(image)

        #Standardize data by dividing by 255
        image = image.astype('float32')/255.0
        data.append(image)
        labels.append(label)

        # Randomly choosing an augmentation operation.
        key = random.choice(list(methods))
        image=methods[key](image)
        data.append(image)
        labels.append(label)

# Generating training dataset.
print("Training data", len(data))

#Shuffle data
data, labels = shuffle(data, labels)
data = np.array(data)
labels = np.array(labels)
train_data=data

Training data 28068
```
Let's try my model with data augmentation.


```python
# Third Prediction
model=cnn_model()
number_epochs=20
batch_size=128

model_fit(model, number_epochs,batch_size)
```

```python
Train on 21051 samples, validate on 7017 samples
Epoch 1/20
21051/21051 [==============================] - 28s 1ms/sample - loss: 1.2831 - acc: 0.4487 - val_loss: 1.0145 - val_acc: 0.5780
Epoch 2/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.9501 - acc: 0.5951 - val_loss: 0.7999 - val_acc: 0.6697
Epoch 3/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.7789 - acc: 0.6966 - val_loss: 0.6473 - val_acc: 0.7525
Epoch 4/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.6329 - acc: 0.7665 - val_loss: 0.6178 - val_acc: 0.7723
Epoch 5/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.5546 - acc: 0.8006 - val_loss: 0.4871 - val_acc: 0.8236
Epoch 6/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.4966 - acc: 0.8195 - val_loss: 0.5334 - val_acc: 0.8062
Epoch 7/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.4575 - acc: 0.8352 - val_loss: 0.4871 - val_acc: 0.8199
Epoch 8/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3987 - acc: 0.8574 - val_loss: 0.4286 - val_acc: 0.8445
Epoch 9/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3701 - acc: 0.8674 - val_loss: 0.4903 - val_acc: 0.8288
Epoch 10/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3255 - acc: 0.8829 - val_loss: 0.4838 - val_acc: 0.8347
Epoch 11/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.2967 - acc: 0.8918 - val_loss: 0.4887 - val_acc: 0.8344
Epoch 12/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.2592 - acc: 0.9085 - val_loss: 0.4946 - val_acc: 0.8333
Epoch 13/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.2484 - acc: 0.9102 - val_loss: 0.5566 - val_acc: 0.8176
Epoch 14/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.2103 - acc: 0.9258 - val_loss: 0.5565 - val_acc: 0.8410
Epoch 15/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.2018 - acc: 0.9270 - val_loss: 0.5551 - val_acc: 0.8340
Epoch 16/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1870 - acc: 0.9330 - val_loss: 0.5289 - val_acc: 0.8407
Epoch 17/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1612 - acc: 0.9432 - val_loss: 0.5778 - val_acc: 0.8384
Epoch 18/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1836 - acc: 0.9368 - val_loss: 0.5725 - val_acc: 0.8465
Epoch 19/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1415 - acc: 0.9489 - val_loss: 0.6176 - val_acc: 0.8371
Epoch 20/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1182 - acc: 0.9572 - val_loss: 0.6233 - val_acc: 0.8427
Runtime: 110212.927397731
```

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m3_acc.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m3_loss.png" alt="">


```python
3000/3000 [==============================] - 2s 510us/sample - loss: 0.6710 - acc: 0.8473
```
![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cm_3.png){: .align-center}


```python
# Construct model
def cnn_model2():
    """function description"""    
    model = Models.Sequential()

    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))

    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))


    model.add(Layers.Flatten())
    model.add(Layers.Dense(256,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(256,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(6,activation='softmax'))

    model.compile(optimizer=Optimizer.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0, amsgrad=True),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])  
    return model
```
So, in the 2nd model I changed the learning rate from 0.0001 to 0.001.

```python
# Fourth Prediction
model=cnn_model2()
number_epochs=60
batch_size=128

model_fit(model, number_epochs,batch_size)
```


```python
Train on 21051 samples, validate on 7017 samples
Epoch 1/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 2.2126 - acc: 0.4135 - val_loss: 1.6589 - val_acc: 0.5766
Epoch 2/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.6342 - acc: 0.5666 - val_loss: 1.4002 - val_acc: 0.6652
Epoch 3/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.4150 - acc: 0.6361 - val_loss: 1.2477 - val_acc: 0.6957
Epoch 4/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.2873 - acc: 0.6771 - val_loss: 1.1649 - val_acc: 0.7173
Epoch 5/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.1817 - acc: 0.7067 - val_loss: 1.0590 - val_acc: 0.7532
[==============================]

Epoch 56/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3517 - acc: 0.9548 - val_loss: 0.6832 - val_acc: 0.8636
Epoch 57/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3496 - acc: 0.9556 - val_loss: 0.6866 - val_acc: 0.8612
Epoch 58/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3583 - acc: 0.9519 - val_loss: 0.6932 - val_acc: 0.8551
Epoch 59/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3458 - acc: 0.9565 - val_loss: 0.6877 - val_acc: 0.8572
Epoch 60/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3387 - acc: 0.9587 - val_loss: 0.6541 - val_acc: 0.8689
Runtime: 99478.285476088
```

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m4_acc.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m4_loss.png" alt="">

```python
3000/3000 [==============================] - 1s 461us/sample - loss: 0.6773 - acc: 0.8687
```

![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cm_4.png){: .align-center}


Let's try another optimizer SGD.

```python
# Construct model
def cnn_model3():
    """function description"""

    model = Models.Sequential()

    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu',input_shape=(100,100,3)))
    model.add(Layers.Conv2D(128,kernel_size=(3,3),activation='relu'))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))

    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Conv2D(256,kernel_size=(3,3),activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.MaxPool2D(pool_size=(3,3)))


    model.add(Layers.Flatten())
    model.add(Layers.Dense(256,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(256,activation='relu',kernel_regularizer=l2(0.001)))
    model.add(Layers.Dropout(0.5))

    model.add(Layers.Dense(6,activation='softmax'))

    model.compile(optimizer=SGD(learning_rate=0.01, momentum=0.8, nesterov=True),
                  loss='sparse_categorical_crossentropy',metrics=['accuracy'])  
    return model
```
REsults:

```python
# Fifth Prediction
model=cnn_model3()
number_epochs=60
batch_size=128

model_fit(model, number_epochs,batch_size)
```
```python
Train on 21051 samples, validate on 7017 samples
Epoch 1/60
21051/21051 [==============================] - 29s 1ms/sample - loss: 2.7490 - acc: 0.3442 - val_loss: 2.3310 - val_acc: 0.5373
Epoch 2/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 2.3351 - acc: 0.5072 - val_loss: 2.3068 - val_acc: 0.5196
Epoch 3/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 2.1780 - acc: 0.5647 - val_loss: 2.0649 - val_acc: 0.6058
Epoch 4/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 2.0855 - acc: 0.5888 - val_loss: 1.9625 - val_acc: 0.6286
Epoch 5/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.9918 - acc: 0.6216 - val_loss: 1.8678 - val_acc: 0.6625
Epoch 6/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 1.8907 - acc: 0.6603 - val_loss: 1.7870 - val_acc: 0.6889
[==============================]
Epoch 57/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3629 - acc: 0.9833 - val_loss: 0.9057 - val_acc: 0.8531
Epoch 58/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3508 - acc: 0.9860 - val_loss: 0.9416 - val_acc: 0.8539
Epoch 59/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3479 - acc: 0.9846 - val_loss: 0.9168 - val_acc: 0.8542
Epoch 60/60
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.3406 - acc: 0.9852 - val_loss: 0.8618 - val_acc: 0.8549
Runtime: 114135.438832549
```
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m5_acc.png" alt="">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m5_loss.png" alt="">

```python
3000/3000 [==============================] - 2s 515us/sample - loss: 0.8744 - acc: 0.8497
```
![image-center]({{ site.url }}{{ site.baseurl }}/images/intel_image/cm_5.png){: .align-center}
