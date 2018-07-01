
from src.settings import ROOT, os

cur_dir = ROOT + "/utils/imgnet_classifier"

CLASSIFIER_DIR = cur_dir + "/classifier"
CLASSIFIER = os.path.join(CLASSIFIER_DIR, "classifier.pkl")

FEATURES_DIR = cur_dir + "/features"
FEATURES = os.path.join(FEATURES_DIR, "train_data.csv")
LABELS = os.path.join(FEATURES_DIR, "train_label.txt")

IMGNET_DIR = cur_dir + "/imgnet"
