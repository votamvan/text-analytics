#!/usr/bin/env python3
import sys
sys.path.append("..")

import os
from pathlib import Path
import html
import re
import numpy as np
import pandas as pd

from sentiment.config import RESOURCE_PATH

DO_DOWNLOAD = False
if DO_DOWNLOAD:
    DATA_PATH = Path('data/')
    DATA_PATH.mkdir(exist_ok=True)
    url_link = "http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    _cmd = "curl -O {}".format(url_link)
    os.system(_cmd)
    _cmd = "tar xzfv ./aclImdb_v1.tar.gz -C {}".format(DATA_PATH)
    os.system(_cmd)

PATH = Path('data/aclImdb/')
re1 = re.compile(r'  +')

def fixup(x):
    x = x.replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>','u_n').replace(' @.@ ','.').replace(
        ' @-@ ','-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x))

CLASSES = ['pos', 'neg', 'unsup']
def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')
print("trn_texts", len(trn_texts), "val_texts", len(val_texts))

col_names = ['labels','text']
np.random.seed(42)
trn_idx = np.random.permutation(len(trn_texts))
val_idx = np.random.permutation(len(val_texts))
trn_texts = trn_texts[trn_idx]
val_texts = val_texts[val_idx]

trn_labels = trn_labels[trn_idx]
val_labels = val_labels[val_idx]
df_trn = pd.DataFrame({'text':trn_texts, 'labels':trn_labels}, columns=col_names)
df_val = pd.DataFrame({'text':val_texts, 'labels':val_labels}, columns=col_names)

df_trn['text'] = df_trn['text'].apply(fixup)
df_val['text'] = df_val['text'].apply(fixup)
df_trn[df_trn['labels']!=2].to_csv(RESOURCE_PATH/'data/imdb-train.csv', header=False, index=False)
df_val.to_csv(RESOURCE_PATH/'data/imdb-test.csv', header=False, index=False)
(RESOURCE_PATH/'data/imdb-classes.txt').open('w', encoding='utf-8').writelines(f'{o}\n' for o in CLASSES)
