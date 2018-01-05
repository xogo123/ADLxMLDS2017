#!/bin/bash

mkdir model_tf
wget http://csie.ntu.edu.tw/~r05922131/model_gan9_bs16_HEclass_original.ckpt.data-00000-of-00001 -O ./model_tf
python generate.py $1
