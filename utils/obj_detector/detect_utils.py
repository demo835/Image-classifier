

from src.settings import *

from src.download import download_and_extract_model
from utils.obj_detector.oid_settings import *
from utils.obj_detector.label_map_utils import string_to_label_map
from utils.obj_detector.draw_obj_utils import draw_results


class OidUtils:
    def __init__(self):
        self.model_dir = OID_MODEL

        self.max_number_of_boxes = 10
        self.minimum_confidence = 0.3

        self.target_width = 700

        # ------------------------ load label data ------------------------------------------------------------
        labeldict_path = OID_LABEL + '/oid_bbox_trainable_label_map.pbtxt.txt'
        if not os.path.exists(labeldict_path):
            sys.stderr.write("no exist oid label data\n")
            return
        else:
            self.label_dicts = self.__load_labeldict(dict_path=labeldict_path)

        # ------------------------ load oid model --------------------------------------------------------------
        data_url = "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_oid_2018_01_28.tar.gz"
        # "http://download.tensorflow.org/models/object_detection/faster_rcnn_inception_resnet_v2_atrous_lowproposals_oid_2018_01_28.tar.gz"
        model_name = (os.path.split(data_url)[1]).split(".tar")[0]
        model_dir = OID_MODEL + "/" + model_name
        model = model_dir + "/frozen_inference_graph.pb"
        if not os.path.exists(model):
            sys.stderr.write("no exist Oid Faster-RCNN model\n")
            download_and_extract_model(data_url=data_url, save_dir=OID_MODEL)
        self.path_to_ckpt = model
        self.__load_model()

    def __load_labeldict(self, dict_path):
        with open(dict_path, 'r') as fp:
            str_label_map = fp.read()
            label_map_dicts = string_to_label_map(str_label_map)
            return label_map_dicts

    def __load_model(self):
        # Load model into memory
        print('loading model {}...'.format(os.path.split(self.path_to_ckpt)[-1]))
        detection_graph = tf.Graph()
        with detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(self.path_to_ckpt, 'rb') as fid:
                serialized_graph = fid.read()
                od_graph_def.ParseFromString(serialized_graph)
                tf.import_graph_def(od_graph_def, name='')

        print('configure model...')
        self.sess = None
        with detection_graph.as_default():
            self.sess = tf.Session(graph=detection_graph)
            self.image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            self.detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            self.detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            self.detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections = detection_graph.get_tensor_by_name('num_detections:0')
        print('initialized.')

    def detect(self, img):
        show_img = img.copy()
        height, width = show_img.shape[:2]

        target_width = int(self.target_width)
        target_height = int(height * target_width / width)

        img = cv2.resize(img, (target_width, target_height))
        image_np_expanded = np.expand_dims(img, axis=0)

        (boxes, scores, classes, num) = self.sess.run(
            [self.detection_boxes, self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np_expanded})

        objs = []
        rects = []
        for i in range(boxes.shape[1]):
            obj_id = int(classes[0][i])
            if scores[0][i] < self.minimum_confidence:
                break

            if obj_id in TARGET_OBJ_IDS:
                y_min, x_min, y_max, x_max = boxes[0][i].tolist()

                cen_pt_x = (x_min + x_max) / 2
                cen_pt_y = (y_min + y_max) / 2

                duplicated = False
                for [x, y, x2, y2] in rects:
                    if x < cen_pt_x < x2 and y < cen_pt_y < y2:
                        duplicated = True
                        break

                if not duplicated:
                    display_name = self.label_dicts[obj_id - 1]["display_name"]
                    label_id = self.label_dicts[obj_id - 1]["id"]
                    if label_id == obj_id:
                        rect = [x_min, y_min, x_max, y_max]
                        rects.append(rect)
                        print("detect object: ", obj_id, display_name)
                        objs.append({'label':  display_name,
                                     'rect': rect,
                                     'confidence': scores[0][i]})

        return objs


if __name__ == '__main__':
    img_path = RAWDATA + "/sample.jpg"
    print(img_path)
    objs = OidUtils().detect(img=cv2.imread(img_path))
    show_img = draw_results(img=cv2.imread(img_path), results=objs)
    cv2.imwrite("result.jpg", show_img)
    cv2.waitKey(0)
