# Object Identification From Image

Based on Convolution Neural Networks

## Summary

### Configuration

- CNN1 to detect the common fish objects from the image (multiple object detector)

- CNN2 to extract the embedded feature from the detected object

- SVM classifier to binary classification(which is positive or negative object)

### CNN For Embedding Feature and SVM Classifier

Location: [utils/imgnet_classifier]

Download the model to [utils/imgnet_classifier/imgnet] from [Inception](https://github.com/tensorflow/models/tree/master/research/inception)

Embedded features of train data: [utils/obj_detector/imgnet] feature [train_data.csv] and labels [train_label.txt]

Classifier: [utils/imgnet_classifier/imgnet/classifier.pkl] which is trained using SVM classification algorithm


### CNN For Object Cropping

Location: [utils/obj_detector]

Download the model from [faster_rcnn_inception_resnet_v2_atrous_oid](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) which was trained with [Dataset](https://github.com/openimages/dataset))

Save and extract the model here [utils/obj_detector/model/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28]

Label data: [utils/obj_detector/model/oid_label_v4]


## Dependencies

- Tensorflow

- OpenCV

- Numpy

- sklearn

install the packages `$ sudo pip install -r requirements.txt`


## Utils

Clone the project
```
 git clone https://github.com/drimyus/Image-classifier.git
```

Run these command lines, for testing and trainig

### Train

```
    python3 src/train.py
```


- prepare the positive and negative images:

(save images on each folder [./data/potivtie] and [./data/negative])

- convert the images to `jpg` format and indexing unique number:

using the functions `convert2JPG()` and `unique_id()` on [src/pre_proc.py]

- collect the embedded features from the raw images [utils/data/]:

`collect_features()` on [utils/imgnet_classifier/features.py]

- train and check the precision of trained model:

`train()` on [utils/imgnet_classifier/train.py]

`check_precision()` on [utils/imgnet_classifier/train.py]

- object(fish) detection based on (pre-trained model)

object detection `OidUtils().detect()` on [utils/obj_detector/detect_utils.py]

bounding rect of detected object `draw_results()` on [utils/obj_detector/draw_obj_utils.py]

### Test


```
python3 src/test.py --file [path of test image e.g. sample.jpg]
```

The `result.jpg` and `result.json` will be created as a result.
