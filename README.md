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
Data used in this project, was taken from [waymo](https://waymo.com/open/) dataset. Data is stored in multiple tensorflow record files.
.tfrecord files are good when storing a large amount data, and improve efficiency when accessing the files.

There are 100 .tfrecord files downloaded for this project from waymo dataset. 
They are split into 77 for training and 20 for evaluation and 3 for testing.
Each tf file holds at minimum image, bounding boxes and classes information.

#### Exploratory Data Analysis
By following Exploratory Data Analysis notebook, it is possible to visualise and observe the dataset.
Observations:
    1. Most of the images are recorded during the day time, with bright sun. That means the brightness of most images is relatively high. There is approximately 1 in 10 images where it is night time. 
    2. Most of the them are taken during at urban areas - cities, towns, suburbs, living areas. 
    3. Most of them have clear weather conditions - high visibility. 
    4. Most of them are clear images with no marks or blur. Only a few has rain or unclear parts.
    5. Images include mostly vehicles or various types, pedestrians, and in rare cases bicycles/motobikes.
  
| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/explore1.png) | 
|:--:| 
| *Example 1 - pedestrians and vehicles, day time* |


| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/explore2.png) | 
|:--:| 
| *Example 2 - vehicles, day time* |


| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/explore5.png) | 
|:--:| 
| *Example 3 - night time* |
 
 
By analysing the data in more detail, it can be identified quantitatively and visually the distribution of class elements(vehicles, pedestrians, cyclists) in the dataset.
 
| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/images/histogram%20of%20classes.png) | 
|:--:| 
| *Histogram of Data Distribution* |
    
### Cross validation

Cross validation is a set of techniques to evaluate the capacity of our model to generalize and alleviate the overfitting challenges. In this course, we will leverage the validation set approach, where we split the available data into two splits:
The recommended way to split the dataset is:
Training: 80% (77 tfrecords)
Validation: 20% (20 tfrecords)

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

When building a ML algorithm, the idea is to create a model which would be suited for real world task.
Due to this, it's vital to have a wide dataset which includes the images from different environment scenarious. 
Such scenarious vary with weather conditions, day time, night time, season of the year.
Therefore, there is technique, where one applies augmentation on image data, to simulate those environments in our training dataset.
This way, the model will generalise well and will not overfit. 

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

It was not possible to upload the whole animation. However, these are screenshots of 2 different animations that were generated with inference_video.py.

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/animation1.png) | 
|:--:| 
| *Animation 1* |

| ![alt text](https://github.com/zbakin/Udacity_Self_Driving_ML/blob/main/experiments/optimisations/images/animation2.png) | 
|:--:| 
| *Animation 2* |
