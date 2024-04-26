#!/bin/bash

mkdir storage
cd "storage"
curl -O http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip
mkdir data
unzip -o uWaveGestureLibrary.zip -d data
cd data
find . -name "*.rar" -exec unrar -y -ad x {} \;
rm *.rar
