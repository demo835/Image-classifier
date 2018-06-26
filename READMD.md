# Object Identification From Image

Based on Neural Network

## Configuration 
- CNN1 to detect the common fish objects from the image (multiple object detector)
- CNN2 to extract the embedded feature from the detected object
- SVM classifier to binary classification(which is positive or negative object)

## Object Cropping CNN1
Model: [faster_rcnn_inception_resnet_v2_atrous_oid](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)
Which was trained with [Dataset](https://github.com/openimages/dataset)

LabelData: []

## Embedding Feature CNN2
Model: [Inception](https://github.com/tensorflow/models/tree/master/research/inception)

LabelData: []

## Dependencies 
Tensorflow
OpenCV
