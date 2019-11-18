---
title: "Tweet Language Identification"
date: 2019-06-06
header:
 image: "/images/lang_iden/nlp2.png"
 teaser: "/images/teaser_images/twitter.jpg"

excerpt: "Language Identification of Tweets"

toc: true
toc_label: " On This Page"
toc_icon: "file-alt"
toc_sticky: true
---




## Introduction
This toy example dataset and project is about utilizing Natural Language Processing in order to make **prediction of language of given tweet.**

**Background Information:** Essentially, given dataset includes diverse tweets with diverse languages. These tweets are gathered between specific time interval.

**My task** is correctly conduct **data analysis and remove outliers**, utilize NLP approaches such as **vectorizing observations,** choosing the best **N-Gram**,forming correct **TF-IDF** (term frequency–inverse document frequency) and finally making classification.

**For this task**, I will only use **Multinomial Naive Bayes** and **Stochastic Gradient Descent Classifier** to have understanding over **effect of n-grams** and **tfidf** parameters.



```python
# Importing fundametal packages
import pandas as pd
import numpy as np
import csv
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## 0. Data Construction and Analysis


```python
# Reading "tweets.json" data. I also added Tweet-ID column name. Later, I'm gonna drop it.
tweets=pd.read_csv('tweets.json',delimiter='\t', sep=',', names=['Tweet-ID'])
test_set=pd.read_csv('labels-test.tsv',delimiter='\t', sep=',',names=['Language','ID'])
```


```python
test_set.head()
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
      <th>Language</th>
      <th>ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>en</td>
      <td>487949847920517120</td>
    </tr>
    <tr>
      <td>1</td>
      <td>id</td>
      <td>487486593112866816</td>
    </tr>
    <tr>
      <td>2</td>
      <td>en</td>
      <td>486698669123846144</td>
    </tr>
    <tr>
      <td>3</td>
      <td>ja</td>
      <td>488764475831377920</td>
    </tr>
    <tr>
      <td>4</td>
      <td>en</td>
      <td>485514378221867008</td>
    </tr>
  </tbody>
</table>
</div>




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
language_codes=pd.read_csv('labels-train+dev.tsv',delimiter='\t',names=['Language','ID'])
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
      <th>Language</th>
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




    Language    object
    ID           int64
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

labaled_test_data=pd.merge(test_set,tweets_clean, on='ID')
```


```python
print(len(labaled_test_data))
```

    13452



```python
labaled_test_data.head()
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
      <th>Language</th>
      <th>ID</th>
      <th>Tweet</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>id</td>
      <td>487486593112866816</td>
      <td>Iyaaa sama sama anti:) \"@Yulianikswnt: SalsaN...</td>
    </tr>
    <tr>
      <td>1</td>
      <td>en</td>
      <td>485514378221867008</td>
      <td>Just really miss my cat tbh."]</td>
    </tr>
    <tr>
      <td>2</td>
      <td>und</td>
      <td>486862316122558464</td>
      <td>http://t.co/gUI2Qf23f3"]</td>
    </tr>
    <tr>
      <td>3</td>
      <td>en</td>
      <td>487489928247668736</td>
      <td>Look soo Mfer foolish"]</td>
    </tr>
    <tr>
      <td>4</td>
      <td>es</td>
      <td>486636216528670720</td>
      <td>-Vamos bra.. -GOL -Podemos empat GOOL -Todavía...</td>
    </tr>
  </tbody>
</table>
</div>



Okay so, let's take a look number of occurences of each tweet with respect to language.


```python
hist=labeled_data["Language"].hist(bins=100, figsize=(30,10))
```


<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/tw/o1.png" alt="">


This histogram is not interpretable so, next step I am gonna sort the occurences of languages considering amount of tweets. In this way, I'll obtain better intuition about my dataset.


```python
occurences=labeled_data["Language"].value_counts()
```


```python
occurences.head()
```




    en     18764
    ja     10421
    es      5978
    und     4835
    id      3038
    Name: Language, dtype: int64



Okay, I obtained sorted dataframe with occurences of languages.


```python
# Creating new dataframe of occurences of languages to be able to make bar plot.
graph_df=pd.DataFrame({'Language':occurences.index, 'Amount':occurences.values})
```


```python
graph_df.head()
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
      <th>Language</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>0</td>
      <td>en</td>
      <td>18764</td>
    </tr>
    <tr>
      <td>1</td>
      <td>ja</td>
      <td>10421</td>
    </tr>
    <tr>
      <td>2</td>
      <td>es</td>
      <td>5978</td>
    </tr>
    <tr>
      <td>3</td>
      <td>und</td>
      <td>4835</td>
    </tr>
    <tr>
      <td>4</td>
      <td>id</td>
      <td>3038</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Creating x-axis labels.
x_labels =graph_df["Language"].values
```


```python
# Creating figure and plotting bar chart with respect to amount of language occurences of Twitter dataset.
plt.figure(figsize=(30,10))
plt.bar(x_labels,graph_df["Amount"])
plt.title("Number of Language Occurences of Tweets", fontsize=30)
plt.xticks(x_labels,fontsize=15,rotation=90)
plt.xlabel('Languages',fontsize=20)
plt.ylabel('Number of Tweets',fontsize=20)
plt.show()
```


<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/tw/o2.png" alt="">

* Obviously, it can be interpreted that dataset **has class imbalance.** It means that each label within dataset is **not equally distributed.**
* Number of tweets on **'en, ja, es, und, id, pt, ar, ru, fr, tr'** are **dominant** within dataset **(Top 10).**
* So, we have more observations (tweets) on those languages whereas we have **very few observations on other languages.**

* In addition these **Top 10 Language** accounts for **95% of overall observations.**


```python
labeled_data.head()
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
      <th>Language</th>
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
  </tbody>
</table>
</div>



Let's drop ID column since we don't need it anymore.


```python
# Dropping ID Columns
labeled_data=labeled_data.drop('ID',1)
labaled_test_data=labaled_test_data.drop('ID',1)
```


```python
# Let's filter particular tweets that we have only observation. First I need to find out which language has only one tweet.

one_obv=graph_df.loc[graph_df["Amount"]==1]

```


```python
one_obv
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
      <th>Language</th>
      <th>Amount</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>58</td>
      <td>mk</td>
      <td>1</td>
    </tr>
    <tr>
      <td>59</td>
      <td>tn</td>
      <td>1</td>
    </tr>
    <tr>
      <td>60</td>
      <td>si</td>
      <td>1</td>
    </tr>
    <tr>
      <td>61</td>
      <td>ar</td>
      <td>1</td>
    </tr>
    <tr>
      <td>62</td>
      <td>ja_LATN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>63</td>
      <td>xh</td>
      <td>1</td>
    </tr>
    <tr>
      <td>64</td>
      <td>ps_LATN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>65</td>
      <td>wo</td>
      <td>1</td>
    </tr>
    <tr>
      <td>66</td>
      <td>ps</td>
      <td>1</td>
    </tr>
    <tr>
      <td>67</td>
      <td>lt</td>
      <td>1</td>
    </tr>
    <tr>
      <td>68</td>
      <td>ml_LATN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>69</td>
      <td>ko_LATN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>70</td>
      <td>ta_LATN</td>
      <td>1</td>
    </tr>
    <tr>
      <td>71</td>
      <td>la</td>
      <td>1</td>
    </tr>
    <tr>
      <td>72</td>
      <td>is</td>
      <td>1</td>
    </tr>
    <tr>
      <td>73</td>
      <td>az</td>
      <td>1</td>
    </tr>
    <tr>
      <td>74</td>
      <td>cy</td>
      <td>1</td>
    </tr>
    <tr>
      <td>75</td>
      <td>dv</td>
      <td>1</td>
    </tr>
    <tr>
      <td>76</td>
      <td>ha</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



So we have different languages with only one tweets. To combat with class imbalance within dataset, I am going to remove observations that has **lower than 10.**


```python
labeled_data=labeled_data.groupby('Language').filter(lambda x: x['Language'].count()>10)
```


```python
print(len(labeled_data))
```

    53336


Number of observation decreased from 53469 to 53336. I did not lose a lot of data but class imbalance took a big hit. Hopefully, I will observe positive classification results during training and testing the dataset.

Now, my dataset is ready to be used in data splitting to obtain training and test dataset, then further use in machine learning.
Let's split observations and labels.


```python
#labeled_data.reset_index(drop=True, inplace=True)
# Splitting observations from labels.
y=labeled_data.iloc[:,0]
X=labeled_data.iloc[:,1]

y_test=labaled_test_data.iloc[:,0]
X_test=labaled_test_data.iloc[:,1]
```


```python
y_test
```




    0         id
    1         en
    2        und
    3         en
    4         es
            ...
    13447     fr
    13448     ja
    13449     es
    13450     ja
    13451    und
    Name: Language, Length: 13452, dtype: object




```python
labeled_data.groupby('Language').size()
```




    Language
    ar          2295
    ar_LATN       12
    ca            22
    de           171
    el            39
    en         18764
    es          5978
    fa            18
    fi            15
    fr           954
    he            27
    hi            16
    hi-Latn       15
    hu            15
    id          3038
    it           339
    ja         10421
    ko           458
    lv            19
    ms           122
    nl           182
    no            11
    pl            93
    pt          2888
    ro            12
    ru           978
    sr            22
    sv            54
    th           465
    tl           320
    tr           669
    uk            16
    und         4835
    ur_LATN       12
    vi            16
    zh-CN         25
    dtype: int64



## 1. Creating Pipeline for Language Processing and Training


```python
# Importing necessary packages for vectorizing and tfidf.
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import LabelEncoder

# Importing pipeline and other machine learning packages.
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score, cross_val_predict
```

### 1.1 Multinomial Naïve Bayes Classifier


```python
# I am using trigram and analyzer char_wb builds up n-grams only from characters inside word boundaries. (user guide)
text_clf_MNB = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
('nb_clf', MultinomialNB())])

text_clf_MNB.fit(X, y)
scores_MNB = cross_val_score(text_clf_MNB, X, y, scoring='accuracy', cv=3)
scores_MNB
```




    array([0.61932872, 0.61889764, 0.61905566])



**3 cross validation** values on training set returns **%61 percent overall accuracy.** The score is not good at all.


```python
# Let's define our accuracy calculator.
def accuracy_calculator(predictor_x, labels):
    """This function accepts machine learning pipeline and labels that needed for testing."""
    correct = 0

    for index, prediction in enumerate(predictor_x):
        if prediction == labels[index]:
            correct +=1

    return print('Accuracy: ', correct/labels.shape[0])
```


```python
# Predictions on test set.

predictor_x=text_clf_MNB.predict(X_test)
labels=y_test

accuracy_calculator(predictor_x, labels)
```

    Accuracy:  0.6210228962236098


* %62 accuracy on test set.
* Results are **terrible** but it is expected due to naive assumption of Multinomial Naive Bayes Classifier.

Let's continue trying different model to see how their behaviors differ.

### 1.1 Stochastic Gradient Descent Classifier


```python
# Let's run Stochastic Gradient Descent Classifier.
text_clf_SGD = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
('SGD', SGDClassifier())])

text_clf_SGD.fit(X, y)
scores_SGD = cross_val_score(text_clf_SGD, X, y, scoring='accuracy', cv=3)
scores_SGD
```




    array([0.91308259, 0.91197975, 0.90894254])




```python
# Predictions on test set.

predictor_x=text_clf_SGD.predict(X_test)
labels=y_test

accuracy_calculator(predictor_x, labels)
```

    Accuracy:  0.9100505501040738


I completed first part of the training. Apparently, **SGD Classifier outperforms MNB model.**

## 2. Hyperparameter Tuning

## 2.1 Multinomial Naive Bayes Model


```python
# Let's start to apply hyperparameter tuning.
from sklearn.model_selection import GridSearchCV

text_clf_tunned_MNB = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(2,3))),
    ('tfidf', TfidfTransformer()),
    ('nb_clf', MultinomialNB())
])

param_grid = {'nb_clf__alpha':[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1. ],
              'nb_clf__fit_prior': ['True', 'False']}

gs_MNB = GridSearchCV(text_clf_tunned_MNB, param_grid, cv=3, n_jobs=-1, verbose=1)
gs_MNB.fit(X, y)
```

    Fitting 3 folds for each of 22 candidates, totaling 66 fits


    [Parallel(n_jobs=-1)]: Using backend LokyBackend with 4 concurrent workers.
    [Parallel(n_jobs=-1)]: Done  42 tasks      | elapsed:  3.6min
    [Parallel(n_jobs=-1)]: Done  66 out of  66 | elapsed:  5.4min finished
    C:\Users\mesut\.conda\envs\TF2\lib\site-packages\sklearn\naive_bayes.py:485: UserWarning: alpha too small will result in numeric errors, setting alpha = 1.0e-10
      'setting alpha = %.1e' % _ALPHA_MIN)





    GridSearchCV(cv=3, error_score='raise-deprecating',
                 estimator=Pipeline(memory=None,
                                    steps=[('vect',
                                            CountVectorizer(analyzer='char_wb',
                                                            binary=False,
                                                            decode_error='strict',
                                                            dtype=<class 'numpy.int64'>,
                                                            encoding='utf-8',
                                                            input='content',
                                                            lowercase=True,
                                                            max_df=1.0,
                                                            max_features=None,
                                                            min_df=1,
                                                            ngram_range=(2, 3),
                                                            preprocessor=None,
                                                            stop_words=None,
                                                            strip_accents=None,...
                                                             smooth_idf=True,
                                                             sublinear_tf=False,
                                                             use_idf=True)),
                                           ('nb_clf',
                                            MultinomialNB(alpha=1.0,
                                                          class_prior=None,
                                                          fit_prior=True))],
                                    verbose=False),
                 iid='warn', n_jobs=-1,
                 param_grid={'nb_clf__alpha': [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6,
                                               0.7, 0.8, 0.9, 1.0],
                             'nb_clf__fit_prior': ['True', 'False']},
                 pre_dispatch='2*n_jobs', refit=True, return_train_score=False,
                 scoring=None, verbose=1)




```python
# Let's print out best cross-validation score.
print("Best cross-validation score: {:.2f}".format(gs_MNB.best_score_))

# Let's print out best parameter of Multinomial Naive Bayes
print(gs_MNB.best_params_)
```

    Best cross-validation score: 0.81
    {'nb_clf__alpha': 0.0, 'nb_clf__fit_prior': 'True'}


## 2.2 Stochastic Gradient Descent Classifier


```python
# Let's start to apply hyperparameter tunning of Stochastic Gradient Descent Classifier
from sklearn.model_selection import GridSearchCV

text_clf_tunned_SGD = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('SGD_clf', SGDClassifier())
])

param_grid = {'SGD_clf__loss':['hinge', 'log'],
              'SGD_clf__penalty': ['l2', 'l1'],
             'SGD_clf__max_iter' :[250,500]}

gs_SGD = GridSearchCV(text_clf_tunned_SGD, param_grid, cv=3, n_jobs=-1, verbose=1)
gs_SGD.fit(X, y)

```


```python
# Let's print best cross-validation score and best parameters of the model.
print("Best cross-validation score: {:.2f}".format(gs_SGD.best_score_))
print("Best Hyperparameters:", gs_SGD.best_params_)
```

Alright, I observed that hyperparameter tuned SGD Classifier outperforms the hyperparameter tuned MNB model.
Therefore, I will **continue** my prediction with **SGD Classifier with tuned hyperparameters.**


```python
best_model = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('SGD_clf', SGDClassifier(loss='hinge',
                              max_iter=2000, penalty='l2', learning_rate='adaptive',eta0=10))
])

best_model.fit(X, y)
```




    Pipeline(memory=None,
             steps=[('vect',
                     CountVectorizer(analyzer='char_wb', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.int64'>, encoding='utf-8',
                                     input='content', lowercase=True, max_df=1.0,
                                     max_features=None, min_df=1,
                                     ngram_range=(1, 3), preprocessor=None,
                                     stop_words=None, strip_accents=None,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, vocabulary=...
                     SGDClassifier(alpha=0.0001, average=False, class_weight=None,
                                   early_stopping=False, epsilon=0.1, eta0=10,
                                   fit_intercept=True, l1_ratio=0.15,
                                   learning_rate='adaptive', loss='hinge',
                                   max_iter=2000, n_iter_no_change=5, n_jobs=None,
                                   penalty='l2', power_t=0.5, random_state=None,
                                   shuffle=True, tol=0.001, validation_fraction=0.1,
                                   verbose=0, warm_start=False))],
             verbose=False)




```python
scores_best_model = cross_val_score(best_model, X, y, scoring='accuracy', cv=3)
scores_best_model
```




    array([0.9139259 , 0.91214848, 0.90888626])




```python
# Prediction on test set.

predictor_x=best_model.predict(X_test)
labels=y_test

accuracy_calculator(predictor_x, labels)
```

    Accuracy:  0.9123550401427297


Wow! Alright, this result is **pretty satisfying.** Now, I want to tweak regularization term to see whether I can **optimize my model even further.** For that, I need to adjust the alpha value in my model parameters. Default **alpha value** is 0.0001.


```python
best_model2 = Pipeline([
    ('vect', CountVectorizer(analyzer='char_wb', ngram_range=(1,3))),
    ('tfidf', TfidfTransformer()),
    ('SGD_clf', SGDClassifier(loss='hinge',
                              max_iter=2000, penalty='l2', learning_rate='adaptive',eta0=5,alpha=0.001))])

best_model2.fit(X, y)
```




    Pipeline(memory=None,
             steps=[('vect',
                     CountVectorizer(analyzer='char_wb', binary=False,
                                     decode_error='strict',
                                     dtype=<class 'numpy.int64'>, encoding='utf-8',
                                     input='content', lowercase=True, max_df=1.0,
                                     max_features=None, min_df=1,
                                     ngram_range=(1, 3), preprocessor=None,
                                     stop_words=None, strip_accents=None,
                                     token_pattern='(?u)\\b\\w\\w+\\b',
                                     tokenizer=None, vocabulary=...
                    ('SGD_clf',
                     SGDClassifier(alpha=0.001, average=False, class_weight=None,
                                   early_stopping=False, epsilon=0.1, eta0=5,
                                   fit_intercept=True, l1_ratio=0.15,
                                   learning_rate='adaptive', loss='hinge',
                                   max_iter=2000, n_iter_no_change=5, n_jobs=None,
                                   penalty='l2', power_t=0.5, random_state=None,
                                   shuffle=True, tol=0.001, validation_fraction=0.1,
                                   verbose=0, warm_start=False))],
             verbose=False)




```python
scores_best_model2 = cross_val_score(best_model2, X, y, scoring='accuracy', cv=3)
scores_best_model2
```




    array([0.82183617, 0.82244094, 0.82216219])




```python
# Prediction on test set.

predictor_x=best_model2.predict(X_test)
labels=y_test

accuracy_calculator(predictor_x, labels)
```

    Accuracy:  0.8151204281891169


So results are getting worse therefore, I will stick with the default regularization value 0.0001. With given model parameters and training methodology, I am able to **correctly predict language** of **%91 of given tweets.**

As my data is unevenly distributed among different languages, there is no point of plotting confusion matrix since heat map will not be intuitive. Rather, I will print out other performance metrics.
