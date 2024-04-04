path = '/data3/anp407/hearing/data/BinauralCuratedDataset/jams'
tpath = path+'/train_org'

# get 1k random from 100k data
import os
import random
import shutil

files = os.listdir(tpath)
random.shuffle(files)
files = files[:100]
for file in files:
    shutil.copytree(tpath+'/'+file, path+'/train/'+file)

# get 100 random from 1k data
vpath = path+'/val_org'
files = os.listdir(vpath)
random.shuffle(files)
files = files[:20]
for file in files:
    shutil.copytree(vpath+'/'+file, path+'/val/'+file)

