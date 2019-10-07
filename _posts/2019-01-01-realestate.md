---
title: "Zurich Real Estate Market"
date: 2018-01-01
header:
 image: "/images/house.png"
 teaser: "/images/house.png"
excerpt: "Data Science and Web Scraping Project on Zurich Real Estate Market."

toc: true
toc_label: " On This Page"
toc_icon: "file-alt"
toc_sticky: true
---
## Introduction

Purpose of this project to generate **web scraping**  codes to scrap one of the biggest real estate website of Switzerland, **comparis.ch,** to gain deeper **insights about Zürich Rental Real Estate Market.**

* **Dataset**: Named as Zurich_Real_Estate_Final.csv, generated with web scraping.
* **Inspiration**: Gain insights about Zürich real estate market.

## Approach

* Project starts with **python code to connect the website,** iterate over relevant 100 pages, gather data and write over csv file.
* Later on, design python code conducts **data cleansing and processing steps** to **prepare data for visualization purposes.**
* Then, clean data is processed within one of the **top visualization software, Tableau,** to generate insight dashboard.
* Clean data is also **mapped into geospatial domain** to provide deeper understanding about districts of Zurich.
* Final dashboard reflects insights about **685 rent post of the website within the scope of Zürich location by January 2019.**


### Web Scraping Code
``` python
##  Webscraping Process
# importing necessary libraries for web scraping
from urllib.request import urlopen as uReq
from bs4 import BeautifulSoup as soup

# creating the file to write data in.
filename="comparis_realestate_data_zürich_only.csv"
f = open(filename, "w", newline='\n')

# writing headers
headers ="title,property_type,living_space,floor,#_of_rooms,street,post_code,city,price\n"
f.write(headers)


# iterating over 100 pages to obtain data from each rental post in website
for i in range(1, 100):

    # connecting to website
    url = 'https://en.comparis.ch/immobilien/marktplatz/zuerich/mieten?'

    # url updater
    url = str(url) + "page=" + str(i)    

    # reading html
    uClient = uReq(url)
    page_html=uClient.read()
    uClient.close()

    #html parsing
    page_soup = soup(page_html, "html.parser")

    # obtain each post
    containers = page_soup.findAll("div", {"class":"content-column columns"})

    # iterating over each container within each page
    for container in containers:
        # trying to obtain relevant data if not assigning not-existing
        try:
            title = container.div.a.text
        except AttributeError:
            title="not-existing"
        try:
            property_type= container.ul.li.text
        except AttributeError:
            property_type="not-existing"
        try:
            living_space_m2=container.ul.li.next_sibling.next_sibling.text
        except AttributeError:
            living_space_m2="not-existing"
        try:   
            floor=container.ul.li.next_sibling.next_sibling.next_sibling.next_sibling.text
        except AttributeError:
            floor="not-existing"
        try:
            number_of_rooms=container.ul.li.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.next_sibling.text
        except AttributeError:
            number_of_rooms="not-existing"
        try:
            post_code=container.find("span", {"class":"street"}).next_sibling.strip()[0:4]
        except AttributeError:
            post_code ="not-existing"
        try:
            city = container.find("span", {"class":"street"}).next_sibling.strip()[5:]
        except AttributeError:
            city="not-existing"
        try:
            street = container.find("span", {"class":"street"}).text
        except AttributeError:
            street="not-existing"
        try:
            price = container.find("div", {"class":"item-price"}).strong.text
        except AttributeError:
            price ="not-existing"

        # writing data to csv file
        f.write(title.replace(",", " ") + "," + property_type.replace(",", " ") + "," +
                    living_space_m2 + "," +
                    floor + "," +
                    number_of_rooms + "," +
                    street.replace(",", " ") + "," +
                    post_code + "," +
                    city + "," + price.replace(",", ".") + "\n")
# closing the file
f.close()
```
So far so good. I am able to iterate over **each post available on website** and **scrap data** into csv file. Next step is cleaning data to make it available for **visualization purposes.**

### Data Cleansing and Preparation for Tableau Visualization
``` python
# transforming data into pandas dataframe
import pandas as pd
import numpy as np

df = pd.read_csv("zurich_main2.csv", encoding="ISO-8859-1", sep=';')

#cleaning data to include only apartments and related property type. Beginning was 990 after cleaning I have 696 observations.
df = df[(df.property_type !='Commercial property') &
        (df.property_type !='Parking space') &
        (df.property_type !='Parking space  garage') &
        (df.property_type !='Single garage') &
        (df.property_type !='Underground garage') &
        (df.property_type !='Building land') &
        (df.property_type !='Hobby room')]

# cleaning no price information posts. After, 685 observations
df = df[(df.price !='On request')]

# cleaning no adress post_code information. 685 observation
df = df[(df.post_code != 'not-existing')]
a =df

# manipulates floor column to replace #_of_rooms column with room values in floor column
a['#_of_rooms'] = np.where((a['floor'] != "Floor 1") & (a['floor'] != "Floor 2") &
  (a['floor'] != "Floor 3") & (a['floor'] != "Floor 4") & (a['floor'] != "Floor 5") &
  (a['floor'] != "Floor 6") & (a['floor'] != "Floor 7") & (a['floor'] != "Floor 8") &
  (a['floor'] != "Floor 9") & (a['floor'] != "Ground floor") & (a['floor'] != "not-existing") &
  (a['floor'] != "Basement"), a['floor'], a['#_of_rooms'])
# manipulates living space to replace #_of_rooms column with room values in living space column
a['#_of_rooms'] = np.where((a['living_space'] == "1 room") | (a['living_space'] == "2 rooms") | (a['living_space'] == "1½ rooms") |
  (a['living_space'] == "2½ rooms") | (a['living_space'] == "3 rooms") | (a['living_space'] == "3½ rooms") |
  (a['living_space'] == "4 rooms") | (a['living_space'] == "4½ rooms") | (a['living_space'] == "5 rooms") |
  (a['living_space'] == "5½ rooms") |(a['living_space'] == "7 rooms") |(a['living_space'] == "9 rooms"), a['living_space'], a['#_of_rooms'])

# manipulates living space to replace floor column with floor values within living space column
a['floor'] = np.where((a['living_space'] == "Floor 1") | (a['living_space'] == "Floor 2") |
  (a['living_space'] == "Floor 3") | (a['living_space'] == "Floor 4") | (a['living_space'] == "Floor 5") |
  (a['living_space'] == "Floor 7") | (a['living_space'] == "Ground floor") |(a['living_space'] == "Basement"),
  a['living_space'], a['floor'])

# assigning null values instead of wrong ones in #_of_rooms column  
a['#_of_rooms'].replace(["Floor 10", "Floor 11", "Floor 12", "Floor 15", "Floor 19", "not-existing", "Floor 14"],
  ['null','null', 'null', 'null', 'null', 'null', 'null'],inplace=True)

# assigning null values instead of wrong ones in floor column
a['floor'] = np.where((a['floor'] !="Floor 1") &(a['floor'] !="Floor 2") & (a['floor'] !="Floor 3") &
        (a['floor'] !="Floor 4") & (a['floor'] !="Floor 5") & (a['floor'] !="Floor 6") & (a['floor'] !="Floor 7") &
        (a['floor'] !="Floor 15") & (a['floor'] !="Floor 8") & (a['floor'] !="Floor 9") & (a['floor'] !="Floor 11") &
        (a['floor'] !="Floor 12") & (a['floor'] !="Floor 19") & (a['floor'] !="Floor 10") & (a['floor'] !="Floor 14") &
        (a['floor'] !="Ground floor") & (a['floor'] !="Basement") , "null", a['floor'])

# assigning null values instead of wrong ones in living_space column
a['living_space'] = np.where((a['living_space'] == "1 room") | (a['living_space'] == "2 rooms") | (a['living_space'] == "1½ rooms") |
  (a['living_space'] == "2½ rooms") | (a['living_space'] == "3 rooms") | (a['living_space'] == "3½ rooms") |
  (a['living_space'] == "5½ rooms") |(a['living_space'] == "4 rooms") | (a['living_space'] == "4½ rooms") |(a['living_space'] == "7 rooms") |  
  (a['living_space'] == "9 rooms") |(a['living_space'] == "Basement") | (a['living_space'] == "5 rooms") |
  (a['living_space'] == "Floor 1") | (a['living_space'] == "Floor 2") | (a['living_space'] == "Floor 3") |
  (a['living_space'] == "Floor 4") | (a['living_space'] == "Floor 5") | (a['living_space'] == "Floor 7") |
  (a['living_space'] == "Ground floor") | (a['living_space'] == "not-existing"), "null", a['living_space'])

# deleting CHF at price column
a['price']=a['price'].str.replace('CHF ', '')
# replacing ½ with .5 at #_of_rooms
a['#_of_rooms']=a['#_of_rooms'].str.replace('½', '.5')
# replacing 6.7, 7, 8, 9 rooms with >=6 rooms at #_of_rooms column
a['#_of_rooms']=a['#_of_rooms'].replace(['6 rooms', '6.5 rooms', '7 rooms', '7.5 rooms', '8.5 rooms', '9 rooms'], ['>=6 rooms',
  '>=6 rooms', '>=6 rooms', '>=6 rooms', '>=6 rooms', '>=6 rooms'])
# deleting m^2 from living_space column value
a['living_space']=a['living_space'].str.replace('m²', '')
#adding district names column
a['district_name']= np.nan

#adding district names to its column accoding to post codes
a['district_name']= np.where((a['post_code'] == "8050"), "Oerlikon", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8046"), "Affoltern", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8052"), "Seebach", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8049"), "Höngg", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8048"), "Altstetten", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8047"), "Albisrieden", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8055"), "Friesenberg", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8063"), "Friesenberg", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8045"), "Alt-Wiedikon", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8002"), "Enge", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8038"), "Wollishofen", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8041"), "Leimbach", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8003"), "Sihlfeld/Werd", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8004"), "Aussersihl", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8005"), "Industriequartier", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8037"), "Wipkingen", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8057"), "Unter/Oberstrass", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8006"), "Wipkingen", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8001"), "Altstadt", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8008"), "Riesbach", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8032"), "Hirslanden", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8053"), "Witikon", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8044"), "Hottingen", a['district_name'])
a['district_name']= np.where((a['post_code'] == "8051"), "Schwamendingen", a['district_name'])

# extracting clean data to csv file
a.to_csv("zurich_main2_clean.csv", index=False, encoding="ISO-8859-1", sep=',')
```
## Results

At the end of the visualization, it is time to express my observations and insights that I got from dataset.

* As one can expect, **old city and surrounding districts** are the more expensive than periphery districts. Specifically, the most expensive districts are **Enge, Altstadt and Hottingen.**
* **The least expensive district is Oerlikon** at the time of the analysis **followed by Wipkingen and Alt-Wiedikon.**
* Generally, **Apartment property type** is **the most common type** that individuals desire to rent out (419 posts). Also Furnished Apartment posts are very common in Oerlikon, Wipkingen Riesbach.
*  In terms of number of rooms, posts mostly gather around 1-room, 2.5-room and 3.5-room rentals.
* Not suprisingly, box plots tell us that **higher the # of rooms, the higher the rents.**
*  Yet, one can observe **intersection** between **2.5-room, 3-room and 3.5-room rent prices.** Therefore, **there is opportunity** to rent out b**igger place with equal amount of rent cost that of smaller place** if **location** is **flexible.**

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/real_estate_project/Dashboard.png" alt="">
