import os
import sys
import numpy as np
import tensorflow as tf
import cv2
import csv
from sklearn.svm import SVC
from sklearn.externals import joblib

cur_dir = os.path.dirname(os.path.realpath(__file__))
ROOT = os.path.dirname(cur_dir)
UTILS = os.path.join(ROOT, "utils")
RAWDATA = os.path.join(ROOT, "data")

IMG_EXTS = ['.png', '.jpeg', '.jpg', '.bmp']
CSV_EXT = [".csv"]


FEATURE_DESC_MODE = "imgnet"  # "resnet"
