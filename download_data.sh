#!/bin/sh

# check if data is there
if [ -d data ]; then
    echo "data directory already present, exiting"
    exit 1
fi

# else create and download
mkdir data
cd data

# download
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz"
wget "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz"

# extract
gunzip *.gz