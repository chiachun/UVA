from ConfigParser import SafeConfigParser
from sklearn.cluster import MeanShift, estimate_bandwidth
import copy
import os
import numpy as np
import pandas as pd
import caffe
import pickle
import cv2
import logging


class Config:
    def __init__(self, *files):
        section_names =['ReadVideo', 'Classification', 'IO', 'Timing']
        parser = SafeConfigParser()
        parser.optionxform = str  # make option names case sensitive
        parser.read(files)
        # Read parameters from configuration files
        for name in section_names:
            items = parser.items(name)
            for (var, val) in items:
                if val.isdigit():
                    self.__dict__.update( {var:int(val)} )
                elif val.replace('.','',1).isdigit():
                    self.__dict__.update( {var:float(val)} )
                else:
                    self.__dict__.update( {var:val} )
       
    def prepare(cfg):
        if not os.path.exists(cfg.csv_dir):
            logging.info('create directory "%s"', cfg.csv_dir)
            os.makedirs(cfg.csv_dir)

        if not os.path.exists(cfg.html_dir):
            logging.info('create directory "%s"', cfg.html_dir)
            os.makedirs(cfg.html_dir)




 # Most of code in this function belongs to Adrian Rosebrock
 # Check http://www.pyimagesearch.com/2014/08/18/
 # skin-detection-step-step-example-using-python-opencv/
def skin_filter(cfg, vd):
    df = pd.read_csv(vd.photo_csv, index_col=0)
    numbers = df.number.tolist()
    notface = []
    for number in numbers:
        lower = np.array([0, 48, 80], dtype = "uint8")
        upper = np.array([13, 255, 255], dtype = "uint8")
        image = cv2.imread('%s/%d.png' % (vd.photo_dir, number), cv2.IMREAD_COLOR)
        converted = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        skinMask = cv2.inRange(converted, lower, upper)

        # apply a series of erosions and dilations to the mask
        # using an elliptical kernel
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        skinMask = cv2.erode(skinMask, kernel, iterations = 2)
        skinMask = cv2.dilate(skinMask, kernel, iterations = 2)

        # blur the mask to help remove noise, then apply the
        # mask to the frame
        skinMask = cv2.GaussianBlur(skinMask, (3, 3), 0)
        skin = cv2.bitwise_and(image, image, mask = skinMask)
        if len(skin.nonzero()[0]) < cfg.min_skin_pixels:
            notface.append(number)
    print '%d/%d are faces' % ( len(df) - len(notface), len(df) )
    df['face']= 1
    df.loc[df.number.isin(notface),'face'] = -99
    df.to_csv(vd.photo_csv)



    
def set_caffe(cfg):
    model = cfg.VGG_model
    weights = cfg.VGG_weights
    if cfg.cpu_caffe:
        caffe.set_mode_cpu()
    net = caffe.Net(model, weights, caffe.TEST)

    # Set up transformer
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_mean('data',np.array([129.1863,104.7624,93.5940]))

    # BGR -> RGB
    transformer.set_channel_swap('data', (2,1,0))
    transformer.set_raw_scale('data',255)
    return net, transformer


def run_caffe(cfg, vd, df):
    # run caffe
    net, transformer = set_caffe(cfg)
    result = extract_feature(df.filename.tolist(), net, transformer, USE_IMAGES=True)
    dfresult = pd.DataFrame(result)
    dfresult['number'] = df.number.tolist()
    df_fc7 = pd.concat([df, dfresult], axis=1, join_axes=[df.index])
    return df_fc7
    


def extract_feature(images, net, transformer, USE_IMAGES = False):
 
    # Extract feature
    num_images = len(images)
    batch_size = 20
    batch_num = (num_images / batch_size)+1
    result = []
    for i in range(0,batch_num):
        start = i*batch_size
        end = (i+1)*batch_size if i<batch_num-1 else num_images
        if start == end: break
        batchimages = images[start:end]
        net.blobs['data'].reshape(len(batchimages),3,224,224)
        if USE_IMAGES == False:
            net.blobs['data'].data[...] =\
                map(lambda x: transformer.preprocess('data',x), batchimages)
        else:
            net.blobs['data'].data[...] =\
                map(lambda x: transformer.preprocess('data',caffe.io.load_image(x)), 
                    batchimages)
        # Run caffe
        out = net.forward()
        X = out['fc7'].tolist()
        result.extend(X) 
    return result


<<<<<<< HEAD
def show_photos_(df,photo_dir, col1, outfile, USE_IMG_PATH=False):
=======
def show_photos_(df,photo_dir, col1, outfile, IMG_PATH_COL_NAME=None):
>>>>>>> correct typo
    header1 ='<!DOCTYPE html> \n <html> \n <head> \n '
    header2 = '<link rel="stylesheet" href="styles.css"> \n </head> \n <body>'
    tailer ='</body> </html>'
    
    f = open(outfile,'w')
    f.write(header1)
    f.write(header2)
    if col1:
        grouped = df.groupby(col1)
        for name, group in grouped:
            f.write("<p> %s %s %d </p>" % (col1, name, len(group)) )
            nums = group.number
            file1s = [ "%s/%d.png" % (photo_dir,n) for n in nums]
            for file1 in file1s:
                f.write('<img alt="not found" src="%s" class="imgshow" />' %file1)
    else:
        file1s = list()
	if IMG_PATH_COL_NAME:
	    file1s = df[IMG_PATH_COL_NAME].tolist()
        else:
            nums = df.number
            file1s = [ "%s/%d.png" % (photo_dir,n) for n in nums]
        for file1 in file1s:
            f.write('<img alt="not found" src="%s" class="imgshow" />' %file1)
    f.write(tailer)
    print "Create %s" % outfile
    f.close()

def show_photos(df_, photo_dir, col1, outfile):
    
    if len(df_) > 3000:
        kvals = np.unique(df_[col1].values)
        kvals_1 = kvals[:len(kvals)/2]
        kvals_2 = kvals[len(kvals)/2:]
        df_1 = df_.loc[df_[col1].isin(kvals_1)]
        df_2 = df_.loc[df_[col1].isin(kvals_2)]
        out1 = outfile.split('.')[0]+'_1'+'.'+outfile.split('.')[1]
        out2 = outfile.split('.')[0]+'_2'+'.'+outfile.split('.')[1]
        show_photos_(df_1, photo_dir, col1, out1)
        show_photos_(df_2, photo_dir, col1, out2)
    else:
        show_photos_(df_, photo_dir, col1, outfile)
        


    
def mean_shift(df, l1, l2, c1name, qt, cluster_all, bin_seeding):
    df1 = df.loc[df[c1name].isin([l1,l2])]
    pccols = [ i for i in range(0,50) ]
    xp = df1[pccols].as_matrix()
    bandwidth = 0
    if l1==l2:
        bandwidth = estimate_bandwidth(xp, quantile=qt)
    else:
        xp1 = df1.loc[df1[c1name]==l1, pccols].as_matrix()
        xp2 = df1.loc[df1[c1name]==l2, pccols].as_matrix()
        bandwidth1 = estimate_bandwidth(xp1, quantile=qt)
        bandwidth2 = estimate_bandwidth(xp2, quantile=qt)
        bandwidth = max(bandwidth1, bandwidth2)
    logging.info("compare (%d, %d) with width=%f", l1, l2, bandwidth)
    ms = MeanShift(bandwidth=bandwidth, cluster_all=cluster_all,
                   bin_seeding=bin_seeding)
    ms.fit(xp)        
    mslabels_unique = np.unique(ms.labels_)
    nc = len(mslabels_unique)
    nl = ms.labels_
    df.loc[df[c1name].isin([l1,l2]), c1name] = df.loc[df[c1name].isin([l1,l2]), c1name]*1000 +nl
    return nc, nl, bandwidth
