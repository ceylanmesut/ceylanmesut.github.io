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
Fundamental challenge of this machine learning project is to create a machine learning model that able to **predict fraudulent credit card transactions** from non-fraduelent ones. From machine learning perspective, fraud transactions are labeled as 1s and non-fraud transactions are labeled as 0s. Therefore, it is **binary classification problem.**

In this project, I'm going to benefit from fundamental classification models from two different categories: probabilistic and non-probabilistic machine learning models. First, I conduct **explanatory data analysis** to better understand data and make decision about my steps. Secondly, I use **four different models on imbalanced dataset** and observe model results. Lastly, I introduce methods for **fight with class imbalance and optimizing the models** for maximum model outcome.

* **Dataset**: Kaggle dataset that anonymized credit card transactions labeled as fraudulent or genuine


* **Inspiration**: Identifying fraudulent credit card transactions.


* **Problem Definition**: Building binary classification models to classify fraud transactions to obtain high AUC and F1 score.


* **Link**: https://www.kaggle.com/mlg-ulb/creditcardfraud


## Approach
* **0.Explanatory Data Analysis**: Understanding the dataset and generating deeper insight.


* **1.Models Againts Imbalanced Dataset**: Analyzing model behaviors against class imbalance.


* **2. Combat with Imbalanced Dataset**: Combating with imbalanced dataset with under-sampling and cost sensitive loss function methods.

## Models
* **Probabilistic Models**:  Logistic Regression and Gaussian Naive Bayes.
* **Non-Probabilistic Models**: Linear Support Vector Machine and Kernelized Support Vector Machine (Polynomial Kernel)






### Web Scraping Code
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

TRAIN_PATH = "../input/seg_train/seg_train/"
TEST_PATH = "../input/seg_test/seg_test/"
```
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


``` python
#Preprocess data
def load_data(path, image_size=100):
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

```python
# Load data
train_data, labels = load_data(TRAIN_PATH, image_size=100)
```

```python
def get_classlabel(class_code):
    labels = {2:'glacier', 4:'sea', 0:'buildings', 1:'forest', 5:'street', 3:'mountain'}  
    return labels[class_code]

f,ax = plt.subplots(3,3)
f.subplots_adjust(0,0,3,3)
for i in range(0,3,1):
    for j in range(0,3,1):
        rnd_number = randint(0,len(train_data))
        ax[i,j].imshow(train_data[rnd_number])
        ax[i,j].set_title(get_classlabel(labels[rnd_number]))
        ax[i,j].axis('off')
```


FOTOGRAFLAR MESELA

``` python
# Construct model
def cnn_model():
    """function description"""

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


<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/real_estate_project/Dashboard.png" alt="">




def model_fit(model, number_epochs, batch_size):
    """This function accepts neural network structure, number of epochs and bathc size as function parameters and train the neural network."""
    start_time = timeit.default_timer()
    # Fit model
    my_model= model
    trained = my_model.fit(train_data,labels,epochs=number_epochs,validation_split=0.25,batch_size=batch_size)

    elapsed = timeit.default_timer()
    print('Runtime:', elapsed)

    plt.plot(trained.history['acc'])
    plt.plot(trained.history['val_acc'])
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    #plt.subplot(2,2,2,frameon=True)
    plt.plot(trained.history['loss'])
    plt.plot(trained.history['val_loss'])
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

```python
model=cnn_model
number_epochs=20
batch_size=32

model_fit(model, number_epochs,batch_size)
```











def make_prediction():
    # Prediction on testing set
    test_images,test_labels = load_data('../input/seg_test/seg_test/')
    test_images = np.array(test_images)
    test_labels = np.array(test_labels)
    model.evaluate(test_images,test_labels, verbose=1)

    # CONF
    predictions = model.predict(test_images)
    pred_labels = np.argmax(predictions, axis = 1)
    frame={'y':test_labels,'y_predicted':pred_labels}
    df = pd.DataFrame(frame, columns=['y','y_predicted'])
    confusion_matrix = pd.crosstab(df['y'], df['y_predicted'],rownames=['True Label'], colnames=['Predicted Label'], margins = False)

    sn.heatmap(confusion_matrix,annot=True,fmt="d",cmap="Blues",linecolor="blue", vmin=0,vmax=500)
    plt.title('Confusion Matrix', fontsize=16)
