# Udacity_Self_Driving_ML

## Project overview
This is the Computer Vision part of the Self-Driving course on Udacity.
The aim of the project is, provided waymo dataset with images of urban environment,
containing cars, pedestrians and cyclists, to train and validate the CNN model.

## Environment set up
Initial project code was taken from:
https://github.com/udacity/nd013-c1-vision-starter.git

### Prerequisites
#### Udacity Workspace
Udacity Workspace environment was used to complete the full project.
#### Docker
Another option is to set up a docker environment with the Dockerfile provided.
For that you would need Ubuntu 20.04 and all dependancies installed.

## Dataset
### Dataset analysis
There are 100 .tfrecord files downloaded for this project from waymo dataset. 
They are split into 77 for training and 20 for evaluation and 3 for testing.
Each tf file holds at minimum image, bounding boxes and classes information.

### Cross validation
The recommended way to split the dataset is:
Training: 77%
Validation: 20%
Testing: 3%

## Training
### Reference experiment
Training starts with running model_main_tf2.py script.
The command used for training:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config
```
The command used for validation:
```
python experiments/model_main_tf2.py --model_dir=experiments/reference/ --pipeline_config_path=experiments/reference/pipeline_new.config --checkpoint_dir=experiments/reference
```

To monitor the training, you can launch a tensorboard instance by running `python -m tensorboard.main --logdir experiments/reference/`

Results:


### Improve on the reference - add augmentations
The command used for training:
```
python experiments/model_main_tf2.py --model_dir=experiments/<experiment_name> --pipeline_config_path=experiments/<experiment_name>/pipeline_new.config
```
The command used for validation:
```
python experiments/model_main_tf2.py --model_dir=experiments/<experiment_name>/ --pipeline_config_path=experiments/<experiment_name>/pipeline_new.config --checkpoint_dir=experiments/<experiment_name>
```

Results:

Reference to other augmentations:
https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto



