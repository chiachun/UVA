import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import pickle

# Switching on PLOT will give you a plot of photos displayed on the first
# and second principle compoenents. Switch off PLOT if dfin contains many photos.
# Otherwise the programm may run out of memory in the process of reading.

PLOT = True

dfin = pd.read_csv('csvs/fc7_000_part_0.csv',index_col=0)
ftcols = ['%d' % i for i in range(0,4096)]  
X = dfin[ftcols] 
pca = PCA(n_components=50)
pca.fit(X)
print '%f percent of variance is explained ' % sum(pca.explained_variance_ratio_)
xp = pca.transform(X)

with open('pca3.pkl', 'w') as pklfile:
    pickle.dump(pca,pklfile)


if PLOT:
    import matplotlib.image as image
    import matplotlib.pyplot as plt
    (fig, ax) = plt.subplots(nrows=1, ncols=1, sharex="col",
                             sharey="row", figsize=(8, 8))
    ax.set_xlabel('pc1')
    ax.set_ylabel('pc2')
    filenames = dfin.filename.tolist()
    for i in range(len(dfin)): 
        im = image.imread(filenames[i])
        ax.imshow(im, aspect='auto', extent=(xp[i,0], xp[i,0]+50 ,
                                             xp[i,1], xp[i,1]+55), zorder=10)
        points0 = ax.scatter(xp[:,0],xp[:,1], s=0)

    plt.savefig("demo_pca.png",bbox_inches='tight')


