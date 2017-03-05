# UVA - Unsupervised Video Analysis

## Required model files
You need to download these files and specify their path in your config.ini to run UVA.

[Haarcascade model](https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_alt2.xml)

[VGG CNN model](http://www.robots.ox.ac.uk/~vgg/software/vgg_face/src/vgg_face_caffe.tar.gz)

[PCA model](http://www.ccfux.url.tw/pca1.pkl)

## Film example 

The film example used in the tutorial can be find [here](https://www.youtube.com/watch?v=NU81m31ig2E)


## Prerequisite packages for UVA
numpy, pandas, caffe, pickle, ConfigParser, 
opencv 3.1, imutils, matplotlib, logging, sklearn


## Start UVA
* Edit config.ini. to set path of inputs and outputs.
* Turn on switches in run_uva.py
* Start a python shell. Execute the script by typing execfile('run_uva.py')

## Outputs
Output csv and html files are named after the scheme "prefix_$num_part_$i.csv (or .html)". 
* photo_$num_part_$i.html displays clustering result
* time_$num_part_$i.html summarizes speaking sessions of speakers.
