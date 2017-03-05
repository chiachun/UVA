#**************************#
# This module is still in construction. 
# It works, but the content is not adequately organized yet.
#**************************#

import pandas as pd
import numpy as np

# Function for convert time in second to h:m:s format
def format_time(tsec):
    hours =  tsec // 3600
    minutes = (tsec % 3600) // 60
    seconds = tsec % 60
    return ['%02d:%02d:%02d' % (h,m,s) for h,m,s in zip(hours,minutes,seconds)]


# Funtion for selecting valid time intervals 
# When two time intervals has a gap < gap_min, 
# they are merged into one interval.
def find_valid_twindows(thist, labelname, tbin, gap_min):
    
    starts = []
    ends = []
    lengths = []

    # Count only bins which have at least one counts
    thist = thist.loc[thist[labelname]>=5].copy(deep=True)

    # If all time windows have no more than 1 counts, return empty lists
    if thist.empty: return starts,ends,lengths

    # Index of the peak bin
    idxmax = thist[labelname].idxmax()

    # All available indices. Note that previous selections may have removed some
    # rows from the dataframe, so the indices may not be continuous. So we need 
    # an index for indices $i to loop through all indices
    idx = thist.index.tolist()
    last_idx = len(idx)-1
    i = 0
    while(i < last_idx):
        length = 1
        starts.append(idx[i])
        j = i + 1
        while(j <= last_idx ):
            # time difference between two adjacent bins
            # if two bins have timediff < $gap_min,
            # we ignore this gap and merge them into one time interval
            timediff = thist.loc[idx[j],'tcenter'] - thist.loc[idx[i],'tcenter']
            if  timediff <= gap_min:
                length = length + int(timediff/tbin) # length in terms of number of bins
                if (j == last_idx): 
                    ends.append(idx[j])
                    lengths.append(length)
                i = j
                j = j + 1
            else:
                ends.append(idx[i])
                lengths.append(length)
                i = j
                break

    # return index
    return starts, ends, lengths

##########################################
# t1, t2 : boundaries of time segments 
##########################################
def time_selection(df, labels, t1, t2, col, photo_dir, gap_min, tbin):
    df['ftime'] = format_time(df.time)

    # Calculate time difference between labeled photos
    gcls = df.groupby(col)
    df['fwdiffs'] = gcls['time'].transform(lambda x: x.diff()) 
    df['bwdiffs'] = gcls['time'].transform(lambda x: x.diff(periods=-1)) 

    # Select1: non-orphan photos are selected  
    df['select1'] = df.apply( lambda x: True if x['bwdiffs']>=-1 or x['fwdiffs'] <=1 else False, axis=1)
    
    df.sort_values(by=[col,'time'])
    # dataframe which contains no orphan photos
    df1 = df.query('select1==True')
    
    ######################
    # Selection 2
    ######################
    # Create histograms using the time window
    tcenters = None
    hists = []
    persons = ['label_%d' % i for i in labels]
    print 'Timing analysis for', persons
    for i in range(0,len(persons)):
        df_km = df1.loc[df1[col]==labels[i]]
        cols = df_km[col].tolist()
        times = df_km.time.tolist()
        filenames = ['%s/%d.png' % (photo_dir,i) for i in df_km.number.tolist()]
        hist, bin_edges = np.histogram(times, bins=range(int(t1),int(t2),tbin))
        tcenters = (bin_edges[:-1] + bin_edges[1:])*0.5
        hists.append(hist)

    # Initialize selection
    min_length = 3
    person_list = []
    start_list = []
    end_list = []
    length_list = []
    dic = {key: value for (key, value) in zip(persons,hists)}
    thist = pd.DataFrame(dic)
    thist['tcenter'] = tcenters

    # Loop through labels
    for person in persons:
        # call find_valid_twindows
        starts, ends, lengths = find_valid_twindows(thist,person,tbin,gap_min)
        if lengths:
            print 'Find valid time intervals for %s' % person
        else:
            print 'No valid time intervals for %s' % person
        for start, end, length in zip(starts, ends, lengths):
            if length >= min_length:
                person_list.append( person )
                start_list.append( start*tbin + t1 )
                end_list.append( end*tbin + t1 )
                length_list.append( length*tbin + t1 )

    # Create dataframe to store selected time intervals
    df_tsel = pd.DataFrame({'person':person_list,'start':start_list, 'end':end_list,'length':length_list })
    
    # Mark photos in selected time intervals
    df['select2'] = False # Initialize select2
    
    # start_list and end_list are in seconds
    for person, start, end in zip(person_list, start_list, end_list):
        print "%s has a valid time interval from %d sec to %d sec" % (person,start,end)
        ith = person.split('_')[1]
        df3 = df.loc[df[col]==int(ith), (col,'time','select2')]
        mapping = np.where(( [start]<=df3.time ) & ( df3.time< [end] ), 'in', 'out')
        g = df3.groupby(mapping).get_group('in')
        df.loc[g.index,'select2'] = True

    # format time in hh:mm:ss  
    df_tsel['fstart']=pd.to_timedelta((df_tsel['start']),unit='s')
    df_tsel['fend']=pd.to_timedelta((df_tsel['end']),unit='s')
    df_tsel = df_tsel.sort_values('start')

    return df, df_tsel, thist


def create_time_table(df, df_tsel, htmlname, col, vd, cfg):

    # Prepare df and df_tsel
    grouped = df.groupby(col)
    
    # accurate start and end time
    accstarts = []
    accends = []
    photos = []
    # insert photos into df_tsel
    for index,row in df_tsel.iterrows():
        person = row['person']
        i = person.split('_')[1]
        start = int( row['start'] )
        end = int( row['end'] )
        group = grouped.get_group(int(i))
        accstart =  group.query('abs(time-%f)<=30' % start).time.min()
        accend =  group.query('abs(time-%f)<=30' % end).time.max()
        accstarts.append(accstart)
        accends.append(accend)
        num_photo = group.query('abs(time-%f)<=30' % start).number.tolist()[1]
        photo = '<img alt="not found" src="%s/%d.png" class="imgshow" onclick="goto(%d)"/>' % (vd.photo_dir, num_photo, accstart)
        photos.append(photo)
    df_tsel['photo']= photos
    df_tsel['accstarts'] = format_time(np.array(accstarts))
    df_tsel['accends'] = format_time(np.array(accends))
    df_tsel = df_tsel[['person','accstarts','accends','photo']]
    df_tsel = df_tsel.sort_values('accstarts')
    df_tsel.columns=['person','start','end','photo']
    

    # Make a html file
    header ='<!DOCTYPE html> \n <html> \n <head> \n'
    css = '<link rel="stylesheet" href="styles.css">  <link rel="stylesheet" href="table.css"> \n'
    js = '<script src="/Users/chiachun/Exp/tagly4/demo/pvideo.js"> </script> \n'
    header2 = '</head> \n <body> '

    lvideo1 = ' <div style="float:left;margin-right:15px;"> <video id="Video1" height="400" controls> '
    lvideo2 = '<source src="%s" type="video/mp4"> </video> </div> \n' % cfg.videoName

    div1 = '<div style="overflow-x:auto;">\n'
    div2 ='</div> </body> </html>'
    pd.set_option('display.max_colwidth', -1)
    f = open(htmlname,'w')
    f.write(header); f.write(css); f.write(js); f.write(header2);
    f.write(lvideo1); f.write(lvideo2); f.write(div1); 
    f.write(df_tsel.to_html(escape=False,index=False))
    f.write(div2)
    f.close()
