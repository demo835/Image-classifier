from src.settings import *


def train_features():
    sys.stdout.write(' >>> generate train data(features) from the raw images \n')
    from utils.imgnet_classifier.features import collect_features
    collect_features()


def train():
    from utils.imgnet_classifier.train import train, check_precision
    train()
    check_precision()


def object_detct():
    img_path = RAWDATA + "/sample.jpg"
    print(img_path)

    from utils.obj_detector.draw_obj_utils import draw_results
    from utils.obj_detector.detect_utils import OidUtils

    objs = OidUtils().detect(img=cv2.imread(img_path))
    show_img = draw_results(img=cv2.imread(img_path), results=objs)

    cv2.imwrite("result.jpg", show_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    pass
