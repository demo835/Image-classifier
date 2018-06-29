from sklearn.svm import SVC
from sklearn.externals import joblib

from src.settings import *


def train_features(feature_mode="imgnet"):
    sys.stdout.write(' >>> generate train data(features) from the raw images \n')
    if not os.path.isdir(RAWDATA):
        sys.stderr.write(" not exist folder for raw image data\n")
        sys.exit(1)

    # --- import the feature descriptor imgnet -------------------------------------------
    if feature_mode == "imgnet":
        from src.feature.imgnet_utils import ImgNetUtils
        emb = ImgNetUtils()
    elif feature_mode == "resnet":
        from src.feature.renet_utils import ResNetUtils
        emb = ResNetUtils()
    else:
        sys.stdout.write("\n unknown descriptor imgnet: {}\n".format(feature_mode))
        sys.exit(0)

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
        fns = [fn for fn in os.listdir(sub_dir_path) if os.path.isfile(path) and os.path.splitext(path)[1].lower() in IMG_EXTS]
        fns.sort()
        for fn in fns:
            path = os.path.join(sub_dir_path, fn)


            try:
                # Extract the feature vector per each image
                feature = emb.get_feature_from_image(path)
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
    save_dir = os.path.join(FEATURES, feature_mode)
    if not os.path.exists(FEATURES):
        os.mkdir(save_dir)
    elif not os.path.isdir(save_dir):
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


def load_feature_and_label(feature_mode):
    sys.stdout.write(' load feature and labels \n')
    sys.stdout.write(' >>> train \n')
    feature_dir = os.path.join(FEATURES, feature_mode)
    feature_data_path = os.path.join(feature_dir, "train_data.csv")
    feature_label_path = os.path.join(feature_dir, "train_label.txt")
    if not os.path.exists(feature_data_path):
        sys.stderr.write(" not exist train data {}\n".format(feature_data_path))
        sys.exit(0)
    if not os.path.exists(feature_label_path):
        sys.stderr.write(" not exist train label {}\n".format(feature_label_path))
        sys.exit(0)

    data = []
    labels = []
    label_names = []
    # --- loading training labels ---------------------------------------------------------------------
    sys.stdout.write(' loading training labels ... \n')
    with open(feature_label_path, 'r') as fp:
        for line in fp:
            line.replace('\n', '')
            label_names.append(line)

    # --- loading training data -----------------------------------------------------------------------
    sys.stdout.write(' loading training data ... \n')
    with open(feature_data_path) as fp:  # for python 2x
        csv_reader = csv.reader(fp, delimiter=',')
        for row in csv_reader:
            feature = [float(row[i]) for i in range(0, len(row) - len(label_names))]
            data.append(np.asarray(feature))

            label_idx = -1
            for i in range(len(label_names)):
                if row[len(feature) + i] == 1.0:
                    label_idx = i
                    break
            if label_idx != -1:
                labels.append(label_names[label_idx])
            else:
                sys.stderr.write(' error on tails for label indicator\n')
                sys.exit(0)
    return data, labels, label_names


def train(feature_mode):
    save_dir = MODELS + "/" + "classifier" + "/" + feature_mode
    if not os.path.exists(MODELS):
        os.mkdir(MODELS)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    classifier_path = save_dir + "/" + "classifier.pkl"

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(feature_mode=feature_mode)

    # --- training -----------------------------------------------------------------------------------
    sys.stdout.write(' training... \n')
    classifier = SVC(C=1.0, kernel='linear', degree=3, gamma='auto', coef0=0.0, shrinking=True, probability=True,
                     tol=0.001, cache_size=200, class_weight='balanced', verbose=False, max_iter=-1,
                     decision_function_shape='ovr', random_state=None)
    classifier.fit(data, labels)
    joblib.dump(classifier, classifier_path)

    sys.stdout.write(' finished the training!\n')


def load_classifier_model(feature_mode):
    classifier_path = MODELS + "/" + "classifier" + "/" + feature_mode + "/" + "classifier.pkl"
    if not os.path.exists(classifier_path):
        sys.stderr.write(" not exist trained classifier {}\n".format(classifier_path))
        sys.exit(0)

    try:
        # loading
        model = joblib.load(classifier_path)
        return model
    except Exception as ex:
        print(ex)
        sys.exit(0)

    
def check_precision(feature_mode):
    # --- load trained classifier imgnet --------------------------------------------------------------
    classifier = load_classifier_model(feature_mode=feature_mode)

    # --- load feature and label data ----------------------------------------------------------------
    data, labels, label_names = load_feature_and_label(feature_mode=feature_mode)

    true_pos = 0
    false_pos = 0
    true_neg = 0
    false_neg = 0
    # --- check confuse matrix ------------------------------------------------------------------------
    sys.stdout.write(' checking the precision... \n')
    for i in range(len(data)):
        feature = data[i]
        feature = feature.reshape(1, -1)

        # Get a prediction from the imgnet including probability:
        probab = classifier.predict_proba(feature)

        max_ind = np.argmax(probab)
        sort_probab = np.sort(probab, axis=None)[::-1]  # Rearrange by size

        if sort_probab[0] / sort_probab[1] < 0.7:
            predlbl = "UnKnown"
        else:
            predlbl = classifier.classes_[max_ind]

        if labels[i] == "positive":
            if predlbl == labels[i]:
                true_pos += 1
            else:
                false_pos += 1
        else:
            if predlbl == labels[i]:
                true_neg += 1
            else:
                false_neg += 1

    sys.stdout.write(' precision result\n')
    sys.stdout.write("positive : (true) {},  (false){}\n".format(true_pos, false_pos))
    sys.stdout.write("negative : (false){},  (true) {}\n".format(false_neg, true_neg))
    total = len(data)
    precision = (true_neg + true_pos) / total
    sys.stdout.write("\nprecision : {} = {} / {}\n".format(precision, true_neg + true_pos, total))


def test(feature_mode, img_path):
    # --- load trained classifier imgnet --------------------------------------------------------------
    classifier = load_classifier_model(feature_mode=feature_mode)

    # --- import the feature descriptor imgnet --------------------------------------------------------
    if feature_mode == "imgnet":
        from src.feature.imgnet_utils import ImgNetUtils
        emb = ImgNetUtils()
    elif feature_mode == "resnet":
        from src.feature.renet_utils import ResNetUtils
        emb = ResNetUtils()
    else:
        sys.stdout.write("\n unknown descriptor imgnet: {}\n".format(feature_mode))
        sys.exit(0)

    # --- extract the feature from image -------------------------------------------------------------
    feature = emb.get_feature_from_image(img_path=img_path)
    feature = feature.reshape(1, -1)

    # --- identify the image label -------------------------------------------------------------------
    probab = classifier.predict_proba(feature)

    max_ind = np.argmax(probab)
    sort_probab = np.sort(probab, axis=None)[::-1]  # Rearrange by size

    if sort_probab[0] / sort_probab[1] < 0.7:
        predlbl = "UnKnown"
    else:
        predlbl = classifier.classes_[max_ind]

    print(predlbl)


if __name__ == '__main__':
    pass
