#!/bin/bash
echo "Checking for CUDA and installing."
# Check for CUDA and try to install.
if ! dpkg-query -W cuda-10-0; then
    curl -O http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    dpkg -i ./cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
    apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
    apt-get update
    apt-get install cuda-10-0 -y
fi
# Enable persistence mode
nvidia-smi -pm 1

# How To Add Swap Space on Ubuntu 18.04
sudo fallocate -l 16G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# install fastai v0.7
sudo apt install -y python3 python3-pip vim tree git git-lfs
pip3 install numpy==1.15.1 bcolz==1.1.2 msgpack==0.5.6 torchtext==0.2.3 fastai==0.7.0 spacy==2.0.18

test -d simple-sentiment || git clone https://github.com/votamvan/simple-sentiment.git
cd simple-sentiment
git pull && git lfs pull