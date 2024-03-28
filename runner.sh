#!/bin/bash

# sample
# extra desc ->  git addess
# nohup ./runner.sh 1 dsrnet_m "Extra description here" > training_{dummy}.log 2>&1 &


## inets = dsrnet_l, dsrnet_m, mdsrnet_l,mdsrnet_m

gpu="$1"
# Set the inet parameter
inet="$2"

# Set the extra description if provided
extra_description="$3"


echo "GPU: $gpu"
# Print the inet parameter
echo "Inet: $inet"

# Print the extra description if provided
if [ -n "$extra_description" ]; then
    echo "Extra Description: $extra_description"
fi

# Run the command with the provided parameters
CUDA_VISIBLE_DEVICES="$gpu", python3 train_sirs.py --inet "$inet" --model dsrnet_model_sirs --dataset sirs_dataset --loss losses --name "$inet" --lambda_vgg 0.01 --lambda_rec 0.2 --if_align --seed 2018 --base_dir "./dataset/reflection-removal/" --nEpochs 1