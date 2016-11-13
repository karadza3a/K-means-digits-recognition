#!/bin/bash

wget --recursive --level=1 --cut-dirs=3 --no-host-directories \
  --directory-prefix=mnist_data --accept '*.gz' http://yann.lecun.com/exdb/mnist/
pushd mnist_data
gunzip *
popd