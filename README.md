# object_detection
Examples and tutorials for object detection algorithms.

# train

Machine: Deep Learning AMI (Ubuntu 18.04) Version 26.0 OS + GPU hardware (https://confluence.check24.de/display/SHFA/AWS+Repository). 

## YOLO

### annotation

This part can be done independent to this repository.

1. Activate virtual environment
2. `pip install labelImg`
3. Launch app:

        labelImg
    or
        
        labelImg [IMAGE_PATH] [PRE-DEFINED CLASS FILE]

    with [PRE-DEFINED CLASS FILE] which has the same form as train/yolo/example/classes.names and [IMAGE_PATH] path to the directory of images.

4. Below the 'Save' in the left column, change save format to yolo.
5. command + w: create a bounding box
6. Either select a class from the dropdown or create a new class. The new class will be added to [PRE-DEFINED CLASS FILE]
7. command + s: save the annotation as a txt file.
8. After finishing the annotation. Make a copy of [PRE-DEFINED CLASS FILE] as classes.names in [IMAGE_PATH]

### training

1. `git clone https://github.com/AlexeyAB/darknet` to get the SoTA repo.

2. In the Makefile, set GPU=1, CUDNN=1, OPENCV=1

3. At line 117 'ifeq ($(GPU), 1) COMMON+= -DGPU -I/usr/local/cuda/include/', replace "/usr/local/cuda/include/" with /usr/local/cuda-11.0/include/

4. Download pre-trained weights depending on the original .cfg file you use. For example, yolov3.cfg and yolov3-spp.cfg will take darknet53.conv.74.

5. Modify the .cfg file. https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects

    - In the yolo layers, modify the 'classes' to the number of classes you have.
    
    - In the convolutional layers, modify the 'filters' to (n_of_classes + 5) * n_of_mask, specified in yolo layers.
    
    - Change line max_batches to (classes*2000, but not less than number of training images and not less than 6000), f.e. max_batches=6000 if you train for 3 classes
    
    - Change line steps to 80% and 90% of max_batches, f.e. steps=4800,5400
    
    - If the GPU memory is not large enough, increase 'subdivisions' in net layers

6. In the training data directory, type the following linux command lines:
    
    - `find $PWD -name '*train*.jpg' > train.txt`
    
    - `find $PWD -name '*val*.jpg' > test.txt`

7. Modify the .data file: 
    
    - specify the locations of the train.txt and test.txt files generated in the last step in train and valid, respectively.
    
    - backup specifies the location of where the trained models will be saved
    
    - specify the location of the classes.names file

8. In the darkent, compile the files: Make -f Makefile.

9. `./darknet detector train <path to .data> <path to .cfg> <path to pretrained weight>`

### Evaluation

Multiple models will be generated during the training, you can use the validation data specified in .data to find the optimized model:

`./darknet detector map <path to .data> <path to .cfg> <path to trained model> > evluation_report.txt`

## Mask RCNN

### annotation

tutorial: https://www.youtube.com/watch?v=-3WVSxNLk_k

software:
    https://www.robots.ox.ac.uk/~vgg/software/via/

1. Add files: add the images you want to annotate.
2. Create new attribute name in the `Attributes/Region Attributes` if you want to create new attribute.
3. You can have as many attributes as you want on the same object.
3. Select the type as dropdown
4. In the `Id/description` section, you can add new classes as you want. The naming of id/description does not matter.
5. In the `Region Shape` section choose the polygon shape and start your annotation.
6. When you finish all your annotations, `Project`/`Save` so you can resume the project in the future.
7. `Annotation`/`Export Annotations (as json)` to export the annotation. Put the annoation in the folder of the image.


### training


#### Requirement:

- folders:
    train/Mask_RCNN
- files:
    -  config/app.config.yml
    -  algorithms/io/\__init\__.py
    -  algorithms/io/metadata_definition.py
    -  algorithms/io/path_definition.py
    -  algorithms.settings
    
1. Setup your environment with environment.yml    
2. Put your training data in the repository
3. Move under the repository 
4. command line:
       
       python -m train/Mask_RCNN/template/template train --dataset=<relative path to training data> --weights=coco --backbone=resnet101 --object_classification=object
