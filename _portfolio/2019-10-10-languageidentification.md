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
Main idea of this project is to successfully **identify different languages of tweets from Twitter API.**

* **Dataset**: Scraped from Twitter API.


* **Inspiration**: Accurately identify language of tweets as accurate as possible.


* **Problem Definition**: Building multiclass classification model to tackle the problem of language identification.



## Approach
* **0.Merging Labels and Observations**: Discovering the dataset and labelling observations.


* **Explanatory Data Analysis**: Understanding the dataset and its distribution.


* **Model Construction**: Constructing **two different machine learning models.**

* **Hyperparameter Tunnig**: Optimizing **hyperparameters** of the ConvNets model to achieve better results.

## Models
* **SGD Classifier**
* **Multinomial Naive Bayes**
* **Utilizing GridSearchCV for both models**


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

