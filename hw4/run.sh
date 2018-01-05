#!/bin/bash

mkdir model_tf
wget https://www.csie.ntu.edu.tw/~r05922131/model_gan9_bs16_HEclass_original.ckpt.data-00000-of-00001 -O ./model_tf/model_gan9_bs16_HEclass_original.ckpt.data-00000-of-00001
wget https://www.csie.ntu.edu.tw/~r05922131/model_gan9_bs16_HEclass_original.ckpt.meta -O ./model_tf/model_gan9_bs16_HEclass_original.ckpt.meta
wget https://www.csie.ntu.edu.tw/~r05922131/model_gan9_bs16_HEclass_original.ckpt.index -O ./model_tf/model_gan9_bs16_HEclass_original.ckpt.index
python generate.py $1
