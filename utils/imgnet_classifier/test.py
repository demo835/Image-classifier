from src.settings import *
from utils.imgnet_classifier.imgnet_settings import *
from utils.imgnet_classifier.imgnet_utils import ImgNetUtils
emb = ImgNetUtils()


def load_classifier_model():
    classifier_path = CLASSIFIER

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


def test(cvimg):
    # --- load trained classifier imgnet --------------------------------------------------------------
    classifier = load_classifier_model()

    feature = emb.get_feature_from_cvMat(cvimg=cvimg)
    feature = feature.reshape(1, -1)

    # Get a prediction from the imgnet including probability:
    probab = classifier.predict_proba(feature)

    max_ind = np.argmax(probab)
    sort_probab = np.sort(probab, axis=None)[::-1]  # Rearrange by size

    if sort_probab[0] / sort_probab[1] < 0.7:
        predlbl = "UnKnown"
    else:
        predlbl = classifier.classes_[max_ind]

    return predlbl
