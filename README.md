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

#### Results

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/reference/images/precision.png) | 
|:--:| 
| *Precision* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/reference/images/recall.png) | 
|:--:| 
| *Recall* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/reference/images/loss.png) | 
|:--:| 
| *Loss* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/reference/images/learning%20rate.png) | 
|:--:| 
| *Learning Rate* |


### Improve on the reference - add augmentations and update optimiser
#### Config file was updated: [pipeline_v2.config](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/pipeline_v2.config)

By experimenting with the model and it's losses, figured out that following improvements make huge impact on accuracy and precision of the model.
Augmentations used:
| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/augmentations2.png) | 
|:--:| 
| *random_adjust_brightness* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/augmentations1.png) | 
|:--:| 
| *random_rgb_to_gray* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/augmentations3.png) | 
|:--:| 
| *random_adjust_contrast* |

Optimiser:
   1. Changed to adam_optimizer
   2. Learning rate changed to - manual_step_learning_rate

#### Results


| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/precision.png) | 
|:--:| 
| *Precision* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/recall.png) | 
|:--:| 
| *Recall* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/loss.png) | 
|:--:| 
| *Loss* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/learning%20rate.png) | 
|:--:| 
| *Learning Rate* |


The command used for training:
```
python experiments/model_main_tf2.py --model_dir=experiments/<experiment_name> --pipeline_config_path=experiments/<experiment_name>/pipeline_new.config
```
The command used for validation:
```
python experiments/model_main_tf2.py --model_dir=experiments/<experiment_name>/ --pipeline_config_path=experiments/<experiment_name>/pipeline_new.config --checkpoint_dir=experiments/<experiment_name>
```

Reference to other augmentations:
https://github.com/tensorflow/models/blob/master/research/object_detection/protos/preprocessor.proto

It was not possible to upload the whole animation. For that reason, I extracted frames on an animation and put them in here:
[animation frames](https://github.com/zbakin/Udacity_Self_Driving_ML/tree/main/experiments/optimisations/animation_frames)

