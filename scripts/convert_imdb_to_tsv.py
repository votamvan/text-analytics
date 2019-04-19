#!/usr/bin/env python3
from pathlib import Path
import html
import re
import os
import numpy as np

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

CLASSES = ['neg', 'pos', 'unsup']
def get_texts(path):
    texts,labels = [],[]
    for idx,label in enumerate(CLASSES):
        for fname in (path/label).glob('*.*'):
            texts.append(fname.open('r', encoding='utf-8').read())
            labels.append(idx)
    return np.array(texts),np.array(labels)

trn_texts,trn_labels = get_texts(PATH/'train')
val_texts,val_labels = get_texts(PATH/'test')