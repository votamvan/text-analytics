#!/bin/bash

source env/bin/activate



pip install bcolz==1.2.1 numpy==1.15.1 msgpack==0.5.6 torchtext==0.2.3 fastai==0.7.0 spacy==2.0.17

pip3 install numpy==1.14.6 msgpack==0.5.6 torchtext==0.2.3 fastai==0.7.0 spacy==2.0.18
python -m spacy download en
cd ./src
nohup python train_en.py &
tail -f nohup.out

sudo -s
pip3 freeze | xargs pip3 uninstall -y



pip3 install numpy==1.14.6 msgpack==0.5.6 torchtext==0.2.3 fastai==0.7.0 spacy==2.0.18

torchtext 0.3.1


pip3 install msgpack==0.5.6 torchtext==0.2.3 fastai==0.7.0 spacy



pip install torchtext==0.2.3 fastai==0.7.0



Successfully installed 
bcolz-1.2.1 
descartes-1.1.0 
fastai-0.7.0 
feather-format-0.4.0 
isoweek-1.3.3 
mizani-0.5.4 
palettable-3.1.1 
pandas-summary-0.0.6 
plotnine-0.5.1 
pyarrow-0.13.0 
sklearn-pandas-1.8.0 
torch-0.3.1 
torchtext-0.2.3