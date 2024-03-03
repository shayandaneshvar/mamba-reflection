#!/bin/bash
docker run -it --runtime=nvidia --name "mamba" -d -v /home/shayan/mamba-reflection:/app kom4cr0/cuda11.7-pytorch1.13-mamba1.1.1:1.1.1

# if other modules were needed just create a new docker image from this

# for opencv
# apt-get update && apt-get install ffmpeg libsm6 libxext6  -y
# pip install requirements -r requirements.txt