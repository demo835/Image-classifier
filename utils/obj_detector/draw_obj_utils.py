from src.settings import *
from utils.obj_detector.oid_settings import CLR_RECT, CLR_TXT


def draw_results(img, results, show_width=None):
    img_h, img_w = img.shape[:2]

    show_img = img.copy()

    for result in results:
        float_rect = result['rect']
        [x, y, x2, y2] = (np.array(float_rect) * np.array([img_w, img_h, img_w, img_h])).astype(np.uint).tolist()
        label = result['label']
        confidence = float(result['confidence'])

        str_label = "{}: {:.1f}%".format(label, confidence * 100)

        show_img = cv2.rectangle(show_img, (x + 10, y + 10), (x2 - 10, y2 - 10), CLR_RECT, 1)
        show_img = cv2.putText(show_img, str_label, (x + 10, y + 10), cv2.FONT_HERSHEY_SIMPLEX, 1, CLR_TXT, 1)

    if show_width is None:
        return show_img
    else:
        return cv2.resize(show_img, (show_width, img_h * show_width // img_w))
