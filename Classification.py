import pickle
import pandas as pd
import sys
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn import preprocessing
from sklearn.cluster import MeanShift, estimate_bandwidth
import numpy as np
from helper import show_photos, mean_shift
import logging

class Classification:
    def __init__(self,cfg,vd,df):
        self.df = df
        self.cfg = cfg
        self.vd = vd
        self.ftcols = ['%d' % i for i in range(0,4096)]  
        self.pca = None
        self.dfp = None
        self.pccols = [ i for i in range(0,50)]
        self.counter = 0
        self.min_counts = cfg.min_counts
        
    def prepare(self):
        with open('%s' % self.cfg.pca_pkl, 'r') as pklfile:
            self.pca = pickle.load(pklfile)
        try:
            self.df = self.df.query('face == 1')
        except:
            print 'Face column not found in the dataframe',
            print 'Treated as not being processed by skin_filter.'
            
        x = self.df[self.ftcols].as_matrix()
        x = preprocessing.scale(x)
        xp = self.pca.transform(x)
        self.dfp = pd.DataFrame(xp)
        self.dfp[['number','time']] = self.df[['number','time']]
        
    def run_kmean(self,n_clusters):
        print 'Run kmean using %d clusters' % n_clusters
        cl1 = KMeans(n_clusters=n_clusters, verbose=0, random_state=1)
        xp = self.dfp[self.pccols].as_matrix()
        labels = cl1.fit_predict(xp)
        centers = cl1.cluster_centers_
        return labels, centers


    def mean_shift_merge(self,col):
        print "======Merge clusters======"
        cfg = self.cfg
        
        # Count merging rounds
        self.counter = 1
        col1 = col + str(self.counter)

        # Show photos before merging
        if cfg.html_pre_merge:
            photo_dir = self.vd.photo_dir
            n_pages = len(self.dfp)/2500 + 1
            kvals = np.unique(self.dfp[col1].values)
            n_kvals = len(kvals)
            shift = n_kvals/n_pages
            p0 = 0
            while (p0 < n_kvals):
                kvals1 = kvals[p0:p0+shift]
                dfp = self.dfp.loc[self.dfp[col1].isin(kvals1)]
                show_photos(dfp, photo_dir, col1,
                            '%s/%s_%d.html' % (cfg.html_dir, cfg.html_pre_merge, p0) )
                p0 = p0 + shift

        # Merge groups
        n2 = 0
        n1 = self.dfp.groupby(col1).ngroups
        while(n2-n1<0 and self.counter < 10):
            col1 = col + str(self.counter)
            col2 = col + str(self.counter+1)
            print "%d round merging" % self.counter
            self.dfp[col2] = self.dfp[col1]
            grouped = self.dfp.groupby(col1)
            grouped = sorted(grouped, key=lambda x: len(x[1]),  reverse=True)
            keys,vals = zip(*grouped)
            keys = list(keys)
            groups = dict(zip(keys,vals))
            i = 0
            while(i<len(groups)):
                k1 = keys[i]
                g1 = groups[k1]
                j = i + 1
                while( j<len(groups) ):
                    k2 = keys[j]
                    g2 = groups[k2]
                    dfg = pd.concat([g1,g2])
                    logging.info("compare classes %d and %d", k1, k2)
                    try:
                        n_cluster, label, __ = mean_shift(dfg, k1, k2, col1,
                                                          0.3, True, False)
                        logging.debug("%d + %d contain %d clusters ",
                                      k1, k2, n_cluster)
                    except:
                        logging.warn("WARNING: mean_shift is not executed")
                        logging.warn("Unexpected error: %s", sys.exc_info()[0])
                        n_cluster = 99
                        
                    # If Meanshift finds that g1 and g2 form one cluster,
                    # merge them  
                    if n_cluster ==1 :
                        print 'Merge %d and %d' % (k1,k2)
                        self.dfp.loc[self.dfp.number.isin(g2.number),col2] = k1
                        g1 = pd.concat([g1,g2])
                        groups.pop(k2)
                        keys.remove(k2)
                        
                    if n_cluster >= 3:
                        groups.pop(k2)
                        keys.remove(k2)
                        
                    j = j + 1
                i = i + 1
            n2 = self.dfp.groupby(col2).ngroups
            n1 = self.dfp.groupby(col1).ngroups
            self.counter = self.counter + 1




    
    def run(self):
        cfg = self.cfg
        self.prepare()
    
        # Run KMean clustering. The resulted cluster centers
        # will be used as seeds for the later MeanShift clustering, which will
        # split the KMean clusters into subclusters if MeanShift find subgroups.  
        n_clusters = len(self.dfp)/cfg.avg_clsize
        labels, centers = self.run_kmean(n_clusters)
        self.dfp['label1'] = labels
        kvals = np.unique(self.dfp.label1.values)
        
        # Use the largest kmean group to estimate MeanShift bandwidth
        idxmax = self.dfp.label1.value_counts().idxmax()
        df_ = self.dfp.loc[self.dfp['label1']==idxmax]
        xp_ = df_[self.pccols].as_matrix()
        bandwidth = estimate_bandwidth(xp_, quantile=0.3)
        
        # run mean shift using centers found by KMmean 
        ms = MeanShift(bandwidth=bandwidth, seeds=centers,
                       cluster_all=True)
        xp = self.dfp[self.pccols].as_matrix()
        ms.fit(xp)        
        mslabels_unique = np.unique(ms.labels_)
        nc = len(mslabels_unique)
        
        # run kmean again using number of clusters found by MeanShift
        labels, centers = self.run_kmean(nc)
        self.dfp['label1'] = labels
        kvals = np.unique(self.dfp['label1'].values)
        print "Classes after the second Kmean: ", kvals
        
        # run mean_shift to analyze KMean clusters 
        # Samples classified as other clusters are assigned new labels
        # New classes whose counts pass the minimum threshold will
        # be kept in the analysis chain, which don't pass will be ignored.
        for kval in kvals:
           __,__, bandwidth = mean_shift(self.dfp, kval, kval, 'label1',
                                         0.3, True, False)
        print "Classification result before merging"
        print "class  counts"
        print self.dfp['label1'].value_counts() 
        # count cut
        cnts = self.dfp['label1'].value_counts()
        passed_cnts = cnts[ cnts>self.min_counts ].index.tolist()
        self.dfp = self.dfp[self.dfp['label1'].isin(passed_cnts)]
        
        self.mean_shift_merge('label')
