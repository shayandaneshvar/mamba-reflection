# Single Image Reflection Removal with Mamba (S6: Structured State-Space Model for Sequences with Selective Scan)
[Full Project Report](https://github.com/shayandaneshvar/mamba-reflection/blob/main/Single%20Image%20Reflection%20Removal%20with%20Mamba.pdf)
## Summary
Replicated the best SOTA Single Image Reflection Removal model (DSRNet), created an image-based version of a SOTA state-space model (Mamba/S6), and replaced attention-based modules of DSRNet with Mamba modules to improve the performance of the model. I also investigated the effect of the Cosine Annealing learning rate schedule and AdamW's weight decay. 

## What I learned

- MAMBA offers a bit of improvement, but not so much, and each Mamba Module has many parameters, making it relatively unstable at this point (early versions).
- CosineAnnealling used with AdamW's Weight Decay is useful.

## Project Details
In this readme file we show the steps required to get the environment ready to run the modules. The project contains lots of files that weren't used, so in Section 1 we mention names of files that are used, and mentioned their purpose. We also indicate which files are completely ours, and which files were changed, and what was the change. Section II covers the steps required to get the project up and running.

Download the full Dataset here: [Dataset](https://drive.google.com/file/d/1hFZItZAzAt-LnfNj-2phBRwqplDUasQy/view?usp=sharing) 

Extra Note: as we discussed, only functions that we wrote should be commented, and so these are found in the files that were (changed) or (ours), indicated in section I. But if anything seemed vague I'm happy to explain it!

### Section I, Description of important files

- dummy.ipynb (ours): mostly code used to play around with Mamba and test various configurations. The only notable part is the last part where I have gathered the logs and used a piece of code to read those again to calculate the average SSIM and PSNR values, and also extract the best epochs of models.

- engine.py (theirs): A piece of code that wrap the complete model and all the parameteres and provies high level train,test, eval api's for easier use

- mamba-container.sh (ours): this file contains the docker command that starts a container with Mamba installed. To use this Nvidia-Container Toolkits should be installed on the system, as well as docker.

- requirements.txt (ours): extra requirements for the project (aside from Mamba and packages installed there).

- runner.sh (ours): This is the file that can be run to train models

- train_sirs.py (theirs)(changed): this is the file that is called by runner.sh and trains and tests the model. Change: Increase Epochs from 20 to 25

- tools and util folders (all theirs): contains bunch of utils that are used while training (and testing) such as different metrics, calculating model size or receptive field, etc. (most files here are useless, like creating an html report, etc. that wasn't used)

- options(theirs) folder contains files that holds program's arguments

- all files in the Data folder (theirs): Dataset files used to read the training set, testing set, and also create that third dataset from VOCDevKit

Files in the models folder (theirs, some changed):

- vgg.py (theirs)-> contains the VGG19 used in the DSFNet and calculating the loss
- losses.py (theirs)-> contains classes that compute the loss

- dsrnet_model_sirs.py(changed) -> contains a wrapper class for the actual network and adds the loss to the network that is in the arch folder. I changed line 173 to add AdamW and 178 to add CosineAnneallingLR, and called the step function in 227-231 to update the scheduler in every iteration.

- __init__.py (theirs) they load the selected model from the arch fodler's __init__.py

arch folder in models(changed):

- lrm.py(theirs) -> contains the lrm module.
- __init__.py (changed)-> high level functions that create the model with a specific number of layers

- dsrnet.py (changed) -> this is where the models are, we have added certain blocks and models including: MuGIMBlock, MUGIM2Block, MuGIMGBlock DualStreamMambaGate, MDSRNet, MXDSRNet, MFeaturePyramidVGG, and MGDSRNet.

-> Code I have added is very easy to read, and so I didn't try to overcomplicate things and just follow pytorch conventions of writing code, hence there wasn't much commenting required. Their code however has some unnecessary wrapping of models, etc. which may be confusing to the reader, which I can happily explain if needed as they don't have comments.

### Section II, Running Guide

Step 1: Environment
Training should be done using Mamba container. Follow steps below:

1 - run the docker command in mamba-container.sh
2 - run docker exec -it mamba bash to enter the container shell
3 - install the files required for opencv to run in the container. the command for installing these are in the same file, commented below the docker command
4 - install the prerequisites for DSRNet, all of these requirements are in the requirements.txt (install with pip)

5 - download the dataset, create a dataset folder right here, and extract the dataset inside it

6 - give running permission to runner.sh with chmod +x runner.sh
Now everything is ready, cd /app where all the files in this repository is mapped via a docker volume, then run the training command in step 2

Step 2: Training

- While in the /app directory of the container, execute the command written below which trains the DSRNet-W on GPU:1 (second GPU) and writes the logs in training.log

  nohup ./runner.sh 1 dsrnet_m "Some helpful comment to remember what this is" > training.log 2>&1 &

- Same command if you want to see the results in standard output (e.g. linux terminal) which will get terminated as soon as logging out of the server:

  ./runner.sh 1 dsrnet_m "Some helpful comment to remember what this is"

The general format of this is:

./runner.sh {gpu index} {model_name} "Some helpful comment to remember what this is"

Where model name can be one of the following: (comes from the function names available in models/arch/__init__.py)

- dsrnet_l
- mdsrnet_l
- mxdsrnet_l
- mgdsrnet_l
- anything that is added in that file

Current version of code uses AdamW and CosineAnnealing, so if any of the networks is selected it will be trained with AdamW and CosineAnnealing. To train with the original setting, i.e. Adam without weight decay and LR Scheduler do the following:

1 - open dsrnet_model_sir.py in models directory
2 - comment out the line 173 and uncomment the line 182
3 - replace line 178 with self.scheduler = None

The results of training will be in the checkpoints directory, where you will find weights for every epoch, the testing results at the end of each epoch, and tensorboard events which can be opened with tensorboard

_CAUTION_: The model produces big weight files, hence each model takes up to 80GB of disk, if you run out of disk, training will stop.

Let me know if you need any of the log files for any of the models, each log file takes at least 50MB and hence I didn't include it and added them to the .gitignore file, but all of them are available on the server.




