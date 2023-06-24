#!/usr/bin/env bash

sudo pip3 uninstall torch y
sudo pip3 uninstall torchvision y
sudo pip3 uninstall torchaudio y
sudo pip3 uninstall mmcv-full y
#sudo pip3 install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
sudo pip3 install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html


#--target=/usr/local/lib/python3.7/site-packages
#export PYTHONPATH="/usr/local/lib/python3.7/site-packages":$PYTHONPATH
sudo pip3 install opencv-python==3.4.6.27 --index-url=https://bytedpypi.byted.org/simple/
sudo pip3 install librosa==0.9.2
sudo pip3 install transformers==4.9.1
sudo pip3 install efficientnet-pytorch==0.7.0
sudo pip3 install seaborn
sudo pip3 install matplotlib==3.3.4
sudo pip3 install pandas==1.1.5
sudo pip3 install xlrd==1.2.0
sudo pip3 install pyarrow==10.0.1
sudo pip3 install timm==0.6.12
sudo pip3 install bytedlaplace
sudo pip3 install bytedeuler~=0.16 -ignore-installed
sudo pip3 install toolz


sudo apt-get -y update
sudo apt-get -y install libsm6
sudo apt-get -y install libxrender1
sudo apt-get -y install libxext-dev
sudo apt-get -y install ffmpeg

#python3 main_v1.py
bash train.sh

