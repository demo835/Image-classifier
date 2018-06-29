from src.settings import *
from utils.imgnet_classifier.imgnet_settings import *

from utils.imgnet_classifier.imgnet_utils import ImgNetUtils
inu = ImgNetUtils()


def collect_features():
    sys.stdout.write(' >>> collect train data(features) from the raw images \n')
    if not os.path.isdir(RAWDATA):
        sys.stderr.write(" not exist folder for raw image data\n")
        sys.exit(1)

    # --- check the raw images ----------------------------------------------------------
    raw_dir = RAWDATA
    sub_dirs = []
    for child in os.listdir(raw_dir):
        child_path = os.path.join(raw_dir, child)
        if os.path.isdir(child_path):
            sub_dirs.append(child)
    sub_dirs.sort()
    labels = sub_dirs

    tails = []
    for i in range(len(sub_dirs)):
        line = np.zeros((len(sub_dirs)), dtype=np.uint8)
        line[i] = 1
        tails.append(line.tolist())
    """
    tails = [[1., 0.],
             [0., 1.]]
    """

    # --- scanning the raw image dir ----------------------------------------------------
    sys.stdout.write("\n scanning folder: {}\n".format(raw_dir))
    features = []
    for sub_dir_name in sub_dirs:
        sub_dir_path = os.path.join(raw_dir, sub_dir_name)

        count = 0
        fns = [fn for fn in os.listdir(sub_dir_path) if
               os.path.isfile(os.path.join(sub_dir_path, fn)) and os.path.splitext(fn)[1].lower() in IMG_EXTS]
        fns.sort()
        for fn in fns:
            path = os.path.join(sub_dir_path, fn)

            try:
                # Extract the feature vector per each image
                feature = inu.get_feature_from_image(path)
                sys.stdout.write("\r" + path)
                sys.stdout.flush()
            except Exception as e:
                print(e)
                continue
            line = feature.tolist()
            line.extend(tails[sub_dirs.index(sub_dir_name)])
            features.append(line)
            count += 1

            # if count > 10:  # for only testing
            #     break

        sys.stdout.write("\nlabel: {}, counts #: {}\n".format(sub_dir_name, count))

    # --- write the train_data.csv file on the same location --------------------------------------
    save_dir = FEATURES_DIR
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    feature_data_path = os.path.join(save_dir, "train_data.csv")
    if sys.version_info[0] == 2:  # py 2x
        with open(feature_data_path, 'wb') as fp:  # for python 2x
            wr = csv.writer(fp, delimiter=',')
            wr.writerows(features)
    elif sys.version_info[0] == 3:  # py 3x
        with open(feature_data_path, 'w', newline='') as fp:  # for python 3x
            wr = csv.writer(fp, delimiter=',')
            wr.writerows(features)

    # write the train_label.txt on the same location
    feature_label_path = os.path.join(save_dir, "train_label.txt")
    with open(feature_label_path, 'w') as fp:
        for label in labels:
            fp.write(label + "\n")

    sys.stdout.write("create the train_data.csv successfully!\n")
    return save_dir


if __name__ == '__main__':
    collect_features()
