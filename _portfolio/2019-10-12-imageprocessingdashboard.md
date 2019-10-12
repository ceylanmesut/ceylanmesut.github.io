---
title: "Image Processing Dashboard with Bokeh"
date: 2019-10-12
header:
 image: "/images/im_pro_dash/im.png"
 teaser: "/images/im_pro_dash/panda.jpg"

excerpt: "Bokeh Dashboard embedded with Salt&Paper Noise, Gaussian and Median Filters"

toc: true
toc_label: " On This Page"
toc_icon: "file-alt"
toc_sticky: true
---

# Image Processing and Interactive Visualization Methods

* **Objective:** Builing **interactive visualization dashboard** where person can add **Salt and Paper noise** to the given image and then apply **2 different image processing filters namely Gaussian and Median Filters** to observe fundamental changes on image and **develop understanding of different filter features and their effects.**
* Some part of the codes in this exercise is pre-provided my part defining function and creating dashboard out of it.


```python
import numpy as np
from numpy.random import rand
from PIL import Image
from bokeh.plotting import figure, curdoc, show
from bokeh.layouts import column, row
from bokeh.models.widgets import Slider, Select
from bokeh.models import ColumnDataSource
import scipy.ndimage as ndimage
from scipy.ndimage import gaussian_filter, median_filter
import random
```


```python
imgnames = ['panda.jpg', 'panda2.jpg']

# create Select widget for image selection
select_image = Select(title="Image:", value="panda.jpg", options=["panda.jpg", "panda2.jpg"])
```


```python
"""
instruction: run with bokeh
bokeh serve --show dva_exc2_skeleton.py
"""

def imageselect(imagename):
    for i in np.arange(2): # change the number here if more than 2 images are included in the "options"
        if imagename in imgnames[i]:
            data_img = Image.open(imgnames[i]).convert('RGBA')
            xdim, ydim = data_img.size
            original = np.flipud(np.asarray(data_img))

            # constructing columndatasource for further processing
            sizesource = ColumnDataSource(data={'size': [xdim, ydim]})
            originalsource = ColumnDataSource(data={'image': [original]})
            imgsource = ColumnDataSource(data={'image': [original]})
            noisesource = ColumnDataSource(data={'image': [original]})
    return sizesource, originalsource, imgsource, noisesource

sizesource, originalsource, imgsource, noisesource = imageselect(select_image.value)


# when selecting different images, update function will be called and all image information should be updated
def update_image(attrname, old, new):
    """This function designed for updating image information"""
    print('update_image(%s)' % new)
    size, origin, image, noise = imageselect(select_image.value)
    sizesource.data.update(size.data)
    originalsource.data.update(origin.data)
    imgsource.data.update(image.data)
    noisesource.data.update(noise.data)
    fig1.image_rgba(image='image', source=noisesource, **img_args)

# Calling function for image selection
select_image.on_change('value', update_image)
```

Alright, now let's create fundamental image processing functions. In this exercise I'm going to use following functions:

* **Salt&Paper**
* **Gaussian Filter**
* **Median Filter functions**


```python
def salt_pepper(noise_density):
    """This function adds Salt and Paper Noise to the given image."""
    # Stating original data source.
    original = originalsource.data['image']
    original = original[0]
    
    # Defining image dimensions.
    xdim, ydim = sizesource.data['size']
    
    # Creating noise image that has the same shape of given image.
    noiseImage = np.zeros(original.shape,np.uint8)
    
    # Iterating over image dimensions.
    for i in range(ydim):
        for j in range(xdim):
            ch = ("Salt","Pepper","nothing")
            no_change= (100 - noise_density)
            rdn = random.choices(ch, weights=[noise_density/2, noise_density/2, no_change])
            if rdn ==['Salt']:
                noiseImage[i][j]= [0, 0, 0, original[i][j][3]]
            elif rdn ==['Pepper']:
                noiseImage[i][j] = [255,255,255,original[i][j][3]]
            else:
                noiseImage[i][j] = [original[i][j][0],original[i][j][1],original[i][j][2],original[i][j][3]]
    
    # Adding noise sources.
    noisesource1 = ColumnDataSource(data={'image': [noiseImage]})
    noisesource2 = ColumnDataSource(data={'image': [noiseImage]})
    return noisesource1, noisesource2


def gauss_filter(noisesource, sigma):
    """This function applies Gaussian Filter on given noisy image and returns filtered image."""
    #Given noise image data.
    noisy = np.asarray(noisesource.data['image'])
    noisy_img = noisy[0]
    
    # Applying gaussian filter to given noisy image and returning gaussian filtered image.
    gauss = ndimage.filters.gaussian_filter(noisy_img, sigma)
    gauss_source = ColumnDataSource(data={'image': [gauss]})
    
    return gauss_source

def med_filter(noisesource, size):
    """This function applies Median Filter on given noisy image and returns filtered image."""
    # Defining noisy data from given image.
    noisy = np.asarray(noisesource.data['image'])
    noisy_img = noisy[0]
    # Applying median filter on noisy image and returning filtered image.
    med = ndimage.median_filter(noisy_img, size) 
    med_source = ColumnDataSource(data={'image': [med]})
    return med_source
```


```python

```

# 2.Slider Creation and Update Function

So far so good. I defined my functions and now I need to create sliders and updates for my interactive dashboard.
Mainly, I'm going to create 3 sliders:

* 1 for Salt&Paper noise adding
* 1 for Gaussian filtering
* 1 for Median filtering

Each slider has 0 to 50 value intervals and each step has 0.1 value change.


```python
# Creating 3 different sliders for my dashboard.
slider_noise = Slider(start=0, end=50, value=0, step=1, title='Noise Density(%):')
slider_gaussian = Slider(start=0, end=5, value=0, step=1, title='Gaussian Filter')
slider_median = Slider(start=0, end=50, value=0, step=1, title='Median Filter')
```

Let's create an update function for noise slider. Since each time I slide the noise density slider, image on the dashboard should be updated according to added noise.


```python
def update_noise(attrname, old, new):
    """This function updates noise slider."""
    # Printing noise amount.
    print('update_noise(%d)' % new)
    # Changing slider noise value.
    ratio = slider_noise.value
    # Updating values.
    salt_pepper_a = salt_pepper(ratio)[0]
    salt_pepper_b= salt_pepper(ratio)[1]
    imgsource.data.update(salt_pepper_a.data)
    noisesource.data.update(salt_pepper_b.data)


# Calling update function for noise slider to change the value of the slider.
slider_noise.on_change('value', update_noise)

def update_gaussian(attrname, old, new):
    """This function updates gaussian filter slider."""
    # Printing noise amount.    
    print('update_gaussian(%f)' % new)
    sigma = slider_gaussian.value
    # Updating values.
    gauss_source = gauss_filter(imgsource, sigma)
    fig1.image_rgba(image='image', source=gauss_source,  **img_args)

# Calling update gaussian function for gaussian filter slider value change.
slider_gaussian.on_change('value', update_gaussian)


def update_median(attrname, old, new):
    """This function updates median filter slider."""
    # Printing noise amount.
    print('update_median(%d)' % new)
    size = slider_median.value
    # Updating values.
    med_source = med_filter(noisesource, size)
    fig1.image_rgba(image='image', source=med_source, **img_args)

# Calling function for median filter slider change
slider_median.on_change('value', update_median)
```

# 3.Figure and Dashboard Creation


```python
# Creating figures and dashboard
xdim, ydim = sizesource.data['size']
fig_args = {'x_range':(0, xdim),'y_range':(0, ydim), 'tools':''}
img_args = {'x':0, 'y':0, 'dw':xdim, 'dh':ydim}


# create figure drawing function for fig1
# this function takes DataColumnSource as input

fig1 = figure(title="Original Image", width=int(1.4*xdim), height=int(1.4*ydim), **fig_args)
fig1.image_rgba(image='image', source=imgsource, **img_args)

sliderColumn = row(column(fig1, row(slider_noise, slider_gaussian, slider_median)), select_image)
layout = sliderColumn

curdoc().add_root(layout)
curdoc().title = "Interactive Dashboard for Noise and Filter Operations"
```


```python

```

<img src="{{ https://ceylanmesut.github.io/classification/.url }}{{ https://ceylanmesut.github.io/classification/.baseurl }}/images/im_pro_dash/dh.png" alt="">
