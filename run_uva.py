from Classification import Classification
from helper import Config, skin_filter, show_photos, run_caffe
from Video import Video
from timing import time_selection, create_time_table
import pickle
import pandas as pd
import numpy as np
import logging

# Switches. You can run every part individually to test 
# and inspect the intermediate products,
# Note that every session needs results of previous sessions.

READVIDEO = False
SPLITFILM = False
RUNCAFFE = False
RUNCLF = True

# Specify the path of configuration file
configfile = "config.ini"

# Register logging object
logging.basicConfig(filename='UVA.log', level=logging.INFO)
logging.info('Analysis Started')

# Switch off pandas chained assignment warning.
# I don't use chain assignment (such as df[a][b]=c),
# but this warning also occur when assigning values to a subset of df
# while this operation actually doesn't have problems.
pd.options.mode.chained_assignment = None


# Read parameters from configuration file 
cfg = Config(configfile)
cfg.prepare()

if READVIDEO:
    vd = Video(cfg)
    vd.read_video()


if SPLITFILM:
    vd = Video(cfg)
    skin_filter(cfg,vd) # face column added
    vd.split_film() # create fc7_$num_part_X.csv and video_$num.pkl
                    # only entries with face==1 are copied into part_.csv
if RUNCAFFE:
    with open('video_%03d.pkl' % cfg.film_num, 'r') as pklfile:
        vd = pickle.load(pklfile)
        
    for i in range(vd.n_segs):
        df = pd.read_csv(vd.fc7_csvs[i], index_col=0)
        df_fc7 = run_caffe(cfg, vd, df)
        df_fc7.to_csv(vd.fc7_csvs[i])
        logging.info('Update %s' ,vd.fc7_csvs[i])

if RUNCLF:
    with open('video_%03d.pkl' % cfg.film_num, 'r') as pklfile:
        vd = pickle.load(pklfile)
        
    for i in range(vd.n_segs):
        df = pd.read_csv(vd.fc7_csvs[i], index_col=0)
        clf = Classification(cfg, vd, df)
        clf.run()
        df_ = clf.dfp
        show_photos(df_, vd.photo_dir, 'label%d' % clf.counter,
                    vd.pic_htmls[i] )
    
        col2 = 'label%d' % clf.counter
        
        # count cut
        cnts = df_[col2].value_counts()
        passed_cnts = cnts[cnts>50].index.tolist()
        df_ = df_[df_[col2].isin(passed_cnts)]
        kvals = np.unique(df_[col2].values)

        # time selection
        df_fc7, df_tsel, df_thist = time_selection(df_, kvals, vd.t1s[i], vd.t2s[i],
                                                   col2, vd.photo_dir, cfg.gap_min,
                                                   cfg.binwidth)
        create_time_table(df_, df_tsel, vd.thtmls[i], col2, vd, cfg)
        df_fc7.to_csv(vd.res_csvs[i])
        df_tsel.to_csv(vd.tsels[i] ) 
        df_thist.to_csv(vd.thists[i] )
