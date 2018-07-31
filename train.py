from src.settings import *
from src.pre_proc import convert2JPG, unique_id
from utils.imgnet_classifier.features import collect_features
from utils.imgnet_classifier.train import train, check_precision


def train_func():
    # 1. preprocessing the train data
    # 1.1 convert all images to jpeg format
    sys.stdout.write(' >>> convert all images to jpeg format \n')
    convert2JPG(os.path.join(RAWDATA + "/" + "positive"))
    convert2JPG(os.path.join(RAWDATA + "/" + "negative"))

    # 1.2 renaming with unique id
    sys.stdout.write(' >>> rename all images with unique id \n')
    unique_id(os.path.join(RAWDATA + "/" + "positive"), tar_prefix="pos")
    unique_id(os.path.join(RAWDATA + "/" + "negative"), tar_prefix="neg")

    # 2. collect the embedded features
    sys.stdout.write(' >>> generate train data(features) from the raw images \n')
    collect_features()

    # 3. train the classifier model
    train()

    # 4. check the accuracy of trained model
    check_precision()


if __name__ == '__main__':
    train_func()
