from src.settings import *
from utils.obj_detector.draw_obj_utils import draw_results
from utils.obj_detector.detect_utils import OidUtils
from utils.imgnet_classifier.test import test

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--file", required=False, default="sample.jpg", help="path to the test image file")
a = parser.parse_args()


def proc(img_path):
    sys.stdout.write("\n\nfile: {}\n".format(img_path))

    img = cv2.imread(img_path)
    img_h, img_w = img.shape[:2]
    objs = OidUtils().detect(img=img)
    show_img = draw_results(img, results=objs)

    json_data = {
        "filename": img_path,
        "size": {"height": img_h, "width": img_w},
    }
    for obj in objs:
        float_rect = obj['rect']
        [x, y, x2, y2] = (np.array(float_rect) * np.array([img_w, img_h, img_w, img_h])).astype(np.uint).tolist()
        crop = img[y:y2, x:x2]
        pred_label = test(cvimg=crop)
        objs.append(
            {
                "rect": {"top": y, "left": x, "bottom": y2, "right": x2},
                "label": pred_label
            })

    json_data['objects'] = objs
    with open("result.json", 'w') as jp:
        json.dump(json_data, jp, indent=2)

    cv2.imwrite("result.jpg", show_img)
    cv2.waitKey(0)


if __name__ == '__main__':
    img_path = a.file

    # _cur_dir = os.path.dirname(os.path.realpath(__file__))
    # img_path = _cur_dir + "/sample.jpg"

    if not os.path.exists(img_path):
        sys.stderr.write("no exist such file {}\n".format(img_path))
        sys.exit(0)
    if os.path.splitext(img_path)[1].lower() not in [".jpg"]:
        sys.stderr.write("not allowed file format {}\n".format(img_path))
        sys.exit(0)

    proc(img_path=img_path)
    sys.stderr.write("Done! result.jpg, result.json\n")
