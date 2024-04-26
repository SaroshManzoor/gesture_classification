#!/bin/bash

mkdir storage
cd "storage"
curl -O http://zhen-wang.appspot.com/rice/files/uwave/uWaveGestureLibrary.zip
mkdir data
unzip -o uWaveGestureLibrary.zip -d data
cd data

# The following loop is generated with the help of Gemini
for file in *.rar; do
    filename="${file%.*}"
    mkdir -p "$filename"
    unrar e "$file" "$filename"
    rm "$file"
done
