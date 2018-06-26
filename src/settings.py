import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import csv

ROOT = os.path.dirname(__file__)
if os.path.split(ROOT)[1] != "image_classifier":
    ROOT = os.path.split(ROOT)[0]

MODELS = os.path.join(ROOT, "models")
FEATURES = os.path.join(ROOT, "features")
RAWDATA = os.path.join(ROOT, "data")

FEATURE_DESC_MODE = "imgnet"  # "resnet"

IMG_EXTS = ['.png', '.jpeg', '.jpg', '.bmp']
CSV_EXT = [".csv"]
