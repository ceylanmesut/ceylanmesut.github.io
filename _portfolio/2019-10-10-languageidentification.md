---
title: "Language Identification"
date: 2019-10-10
header:
 image: "/images/lang_iden/nlp2.png"
 teaser: "/images/teaser_images/twitter.jpg"

excerpt: "Language Identification from Tweets"

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


## Data Exploration

Let's load necessary packages.



```python
# Importing fundametal packages
import pandas as pd
import numpy as np
import csv
```


```python
# Reading "tweets.json" data. I also added Tweet-ID column name. Later, I'm gonna drop it.
tweets=pd.read_csv('tweets.json',delimiter='\t', sep=',', names=['Tweet-ID'])
```


```python
#Checking top 10 lines from head of data.
tweets.head(10)         
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet-ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>["483885347374243841","اللهم أفرح قلبي وقلب من...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>["484023414781263872","إضغط على منطقتك يتبين ل...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>["484026168300273664","اللَّهٌمَّ صَلِّ وَسَلِ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>["483819942878650369","@Dinaa_ElAraby اها يا ب...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>["483793769079123971","• افضل كتاب قرأته هو : ...</td>
    </tr>
    <tr>
      <td>5</td>
      <td>["483934868070350849","@hudc7721 انتظري اجل \n...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>["483863369972473856","(وإن تجهر بالقول فإنه ي...</td>
    </tr>
    <tr>
      <td>7</td>
      <td>["483871567311413248","ﺧﻟك ﻋزﯾز آﻟﻧﻓس ﻟۈ ﮪﻣۈﻣك...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>["483931429902884864","عشان الجنّة أجمل ؟  الل...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>["483773756897124352","توجيه كيفية تثبيت البرا...</td>
    </tr>
  </tbody>
</table>
</div>



Apparently, pandas dataframe consists of lists. Each row represented as lists and tweet-id and tweet itsel are represented as list elements. At this point, I need to seperate these two different values into two different columns.


```python
# As tweet-ids are 17 character long unique numbers, I filter them into new column that I named as ID. 
tweets['ID']=tweets["Tweet-ID"].str[2:20]

# 2nd of the list element is tweet itself. Therefore, I allocated 2nd element into Tweet column.
tweets['Tweet']=tweets["Tweet-ID"].str[23:]
```


```python
# Let's check seperated data. 
tweets.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Tweet-ID</th>
      <th>ID</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>["483885347374243841","اللهم أفرح قلبي وقلب من...</td>
      <td>483885347374243841</td>
      <td>اللهم أفرح قلبي وقلب من أحب وأغسل أحزاننا وهمو...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>["484023414781263872","إضغط على منطقتك يتبين ل...</td>
      <td>484023414781263872</td>
      <td>إضغط على منطقتك يتبين لك كم يتبقى من الوقت عن ...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>["484026168300273664","اللَّهٌمَّ صَلِّ وَسَلِ...</td>
      <td>484026168300273664</td>
      <td>اللَّهٌمَّ صَلِّ وَسَلِّمْ عَلىٰ نَبِيِّنَآ مُ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>["483819942878650369","@Dinaa_ElAraby اها يا ب...</td>
      <td>483819942878650369</td>
      <td>@Dinaa_ElAraby اها يا بيبي والله اتهرست علي تو...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>["483793769079123971","• افضل كتاب قرأته هو : ...</td>
      <td>483793769079123971</td>
      <td>• افضل كتاب قرأته هو : أمي (ابراهام لنكولن)\n🌹...</td>
    </tr>
  </tbody>
</table>
</div>



Now, I obtained column-wise seperated observations. 
One column belongs to Tweet ID and other column belongs to Tweet itself. This approach will ease the process of labelling data.

Let's drop the main column that has two list elements and finalize this dataset.


```python
# Dropping main data column as I seperated two elements into different columns.
tweets_clean=tweets.drop('Tweet-ID',1)
```


```python
tweets_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>483885347374243841</td>
      <td>اللهم أفرح قلبي وقلب من أحب وأغسل أحزاننا وهمو...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>484023414781263872</td>
      <td>إضغط على منطقتك يتبين لك كم يتبقى من الوقت عن ...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>484026168300273664</td>
      <td>اللَّهٌمَّ صَلِّ وَسَلِّمْ عَلىٰ نَبِيِّنَآ مُ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>483819942878650369</td>
      <td>@Dinaa_ElAraby اها يا بيبي والله اتهرست علي تو...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>483793769079123971</td>
      <td>• افضل كتاب قرأته هو : أمي (ابراهام لنكولن)\n🌹...</td>
    </tr>
  </tbody>
</table>
</div>




```python
tweets_clean.dtypes
```




    ID       object
    Tweet    object
    dtype: object




```python
# Changing data type of ID into integer.
tweets_clean["ID"]=tweets_clean["ID"].astype(np.int64)
```

Alright. Now, I have better structured dataset. It's time to discover label dataset and map the labels to the tweets dataset. I'm adding two column names easly as Language and ID.


```python
language_codes=pd.read_csv('labels-train+dev.tsv',delimiter='\t',names=['Languge','ID'])
```


```python
language_codes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Languge</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ar</td>
      <td>483762194908479488</td>
    </tr>
    <tr>
      <td>1</td>
      <td>ar</td>
      <td>483762916097654784</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ar</td>
      <td>483764828784582656</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ar</td>
      <td>483765526683209728</td>
    </tr>
    <tr>
      <td>4</td>
      <td>ar</td>
      <td>483768342315282432</td>
    </tr>
  </tbody>
</table>
</div>




```python
language_codes.dtypes
```




    Languge    object
    ID          int64
    dtype: object



Alright, everything seems fine now. It is important to have same data types for both columns. Since ID columns of language_codes dataset and Tweets dataset are int64, I can start mapping language label mapping process.


```python
print("Number of rows of Train+Dev Dataset:",len(language_codes))
print("Number of rows of Tweet Dataset:",len(tweets_clean))
```

    Number of rows of Train+Dev Dataset: 96414
    Number of rows of Tweet Dataset: 66921
    

Essentially, there are more labels than number of tweets. So, when dataset is obtained from Tweeter API, some of the tweets are deleted.

Let's take a look at how many observations are matching and how many of them are not matching. I am looking forward to obtain the same results when I conduct "inner join operation on ID key".


```python
tweets_clean['ID'].isin(language_codes['ID']).value_counts()
```




    True     53469
    False    13452
    Name: ID, dtype: int64



Inner join operation incudes two different dataset match on common "key", in my case "ID columns" of both datasets, and rest of the unmatching observation of both dataset is removed. Therefore, final dataset only includes observations matching on common key.


```python
# Merging operation. Left dataset is language_codes and right dataset tweets_clean. 
labeled_data=pd.merge(language_codes,tweets_clean, on='ID')
```


```python
print(len(labeled_data))
```

    53469
    


```python
labeled_data.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Languge</th>
      <th>ID</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ar</td>
      <td>483762194908479488</td>
      <td>@a7medmagdy11 كان قاعد معايا وقت الماتش و قال ...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>ar</td>
      <td>483762916097654784</td>
      <td>يا من أناديها ويخنقني البكاء \nويكاد صمت الدمع...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ar</td>
      <td>483765526683209728</td>
      <td>فيه فرق بين اهل غزة اللى مطحونين من ناحيتين وب...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ar</td>
      <td>483768342315282432</td>
      <td>ﻋﻦ ﺍﻟﻠﺤﻈﺔ اﻟﺤﻠﻮﺓﺓ ﺍﻟﻠﻲ ﺑﺘﻐﻤﺾ ﻓﻴﻬﺎ ﻋﻴﻨﻴﻚ ﺑﺘﻔﻜﺮ ...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>ar</td>
      <td>483770765985083392</td>
      <td>يا ابو سلو عرفتني"]</td>
    </tr>
    <tr>
      <td>5</td>
      <td>ar</td>
      <td>483770900127285248</td>
      <td>ب50 ريال أكفل معتمر في رمضان ، ولك بإذن الله م...</td>
    </tr>
    <tr>
      <td>6</td>
      <td>ar</td>
      <td>483773756897124352</td>
      <td>توجيه كيفية تثبيت البرامج الثابتة ROM التحميل ...</td>
    </tr>
    <tr>
      <td>7</td>
      <td>ar</td>
      <td>483780914452111360</td>
      <td>{وأنه هو أغنى وأقنى} [النجم:48]\nhttp://t.co/i...</td>
    </tr>
    <tr>
      <td>8</td>
      <td>ar</td>
      <td>483782119827582977</td>
      <td>اللهم قدر لنا الفرح بكل اشكاله ، انت الكريم ال...</td>
    </tr>
    <tr>
      <td>9</td>
      <td>ar</td>
      <td>483782693868433408</td>
      <td>#غزه_تحت_القصف\n\nداعش أخواني حيل عندكم بالمدن...</td>
    </tr>
    <tr>
      <td>10</td>
      <td>ar</td>
      <td>483790646176935936</td>
      <td>{يعلمون ظاهرا من الحياة الدنيا وهم عن الآخرة ه...</td>
    </tr>
    <tr>
      <td>11</td>
      <td>ar</td>
      <td>483793769079123971</td>
      <td>• افضل كتاب قرأته هو : أمي (ابراهام لنكولن)\n🌹...</td>
    </tr>
    <tr>
      <td>12</td>
      <td>ar</td>
      <td>483796979063848960</td>
      <td>ولأنّهُم مَلائِكَةٌ صِغار..نَعْشَقُ اتِكاءة رؤ...</td>
    </tr>
    <tr>
      <td>13</td>
      <td>ar</td>
      <td>483802793459736576</td>
      <td>خُلاصة الحُب هي تُفكر بقلبهآ وهو يُفكر بعقلهِ ...</td>
    </tr>
    <tr>
      <td>14</td>
      <td>ar</td>
      <td>483807295134900224</td>
      <td>جميل آن يفهمك منَ تحبب ويخآفَ عليك و يغآر عليك...</td>
    </tr>
    <tr>
      <td>15</td>
      <td>ar</td>
      <td>483811483906625536</td>
      <td>حتى الندم على المعصيه تؤجر عليه - سبحانك يالله...</td>
    </tr>
    <tr>
      <td>16</td>
      <td>ar</td>
      <td>483819942878650369</td>
      <td>@Dinaa_ElAraby اها يا بيبي والله اتهرست علي تو...</td>
    </tr>
    <tr>
      <td>17</td>
      <td>ar</td>
      <td>483824293952749568</td>
      <td>(لا يقاتلونكم جميعا إلا في قرى محصنة أو من ورا...</td>
    </tr>
    <tr>
      <td>18</td>
      <td>ar</td>
      <td>483850786208612352</td>
      <td>طبت منك نهائياً, بس كلي رجاء ما ترجع من جديد."]</td>
    </tr>
    <tr>
      <td>19</td>
      <td>ar</td>
      <td>483863369972473856</td>
      <td>(وإن تجهر بالقول فإنه يعلم السر وأخفى) [طه:7]\...</td>
    </tr>
  </tbody>
</table>
</div>




```python
language_codes  =language_codes.set_index('ID')
tweets_clean=tweets_clean.set_index('ID')
```


```python
tweets_clean.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ID</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>483885347374243841</td>
      <td>اللهم أفرح قلبي وقلب من أحب وأغسل أحزاننا وهمو...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>484023414781263872</td>
      <td>إضغط على منطقتك يتبين لك كم يتبقى من الوقت عن ...</td>
    </tr>
    <tr>
      <td>2</td>
      <td>484026168300273664</td>
      <td>اللَّهٌمَّ صَلِّ وَسَلِّمْ عَلىٰ نَبِيِّنَآ مُ...</td>
    </tr>
    <tr>
      <td>3</td>
      <td>483819942878650369</td>
      <td>@Dinaa_ElAraby اها يا بيبي والله اتهرست علي تو...</td>
    </tr>
    <tr>
      <td>4</td>
      <td>483793769079123971</td>
      <td>• افضل كتاب قرأته هو : أمي (ابراهام لنكولن)\n🌹...</td>
    </tr>
  </tbody>
</table>
</div>




```python
language_codes.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Languge</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>ar</td>
      <td>483762194908479488</td>
    </tr>
    <tr>
      <td>1</td>
      <td>ar</td>
      <td>483762916097654784</td>
    </tr>
    <tr>
      <td>2</td>
      <td>ar</td>
      <td>483764828784582656</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ar</td>
      <td>483765526683209728</td>
    </tr>
    <tr>
      <td>4</td>
      <td>ar</td>
      <td>483768342315282432</td>
    </tr>
  </tbody>
</table>
</div>


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
[==============================]
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

## **Data Augmentation**
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

#Defining augmentation operations.
def horizontal_flip(image):
    """Flips the given image horizontally"""
    return image[:, ::-1]

def up_side_down(image):
    return np.rot90(image, 2)

# Defining augmentation methods.    
methods={'h_flip':horizontal_flip,'u_s_d':up_side_down}
# Defining data and label lists to append images into.
data = []
labels = []
# Setting the path of data.
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
After data augmentation process, I doubled my training data amount **from 14k to 28k images.** Let's try my model with data augmentation.


```python
# Third Prediction
model=cnn_model()
number_epochs=20
batch_size=128
model_fit(model, number_epochs,batch_size)
```
Now model trains with 21k images and validates with 7k images.

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
[==============================]
Epoch 18/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1836 - acc: 0.9368 - val_loss: 0.5725 - val_acc: 0.8465
Epoch 19/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1415 - acc: 0.9489 - val_loss: 0.6176 - val_acc: 0.8371
Epoch 20/20
21051/21051 [==============================] - 27s 1ms/sample - loss: 0.1182 - acc: 0.9572 - val_loss: 0.6233 - val_acc: 0.8427
Runtime: 110212.927397731
```
**Overfitting** problem is **still existing** in the model.

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m3_acc.png" alt="AA">
<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/intel_image/m3_loss.png" alt="AA">


```python
3000/3000 [==============================] - 2s 510us/sample - loss: 0.6710 - acc: 0.8473
```
**No substantial improvement** on model accuracy.

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

## Results

At the end of this project, I tried different approaches to be able to classify intel landscape image dataset as accurate as possible. I used **ConvNets** to tackle this problem. To **avoid overfitting** I utilized **L2 Regularization (Gaussian Prior/Ridge) along with dropout and data augmentation.**

Finally, designed deep learning model is able to **classify landscape images with around %87 success rate.**

As next step, one can change deep learning structure, making model deeper or shallower, including average or sum pooling approaches. In addition, one can also try out different optimizer such as Stochastic Gradient Descent or Adagrad with optimizing learning rate and epochs.
