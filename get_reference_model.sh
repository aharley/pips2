#!/bin/bash

echo "downloading the model from dropbox..."
wget https://www.dropbox.com/scl/fi/czdlt2zc2ji2b7zd0pvoe/reference_model.tar.gz?rlkey=56ebq4g5dk01kyq8kuismev14 \
  -O reference_model.tar.gz

echo "extracting from tar..."
tar -xvf reference_model.tar.gz

echo "deleting the tar..."
rm -v reference_model.tar.gz

echo "done"
