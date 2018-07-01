import tarfile
from six.moves import urllib

from src.settings import *
from utils.imgnet_classifier.imgnet_settings import *


class ImgNetUtils:
    def __init__(self):
        self.model_dir = IMGNET_DIR

        self.__create_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=config)
        self.softmax_tensor = self.sess.graph.get_tensor_by_name('pool_3:0')

        sys.stdout.write("...init mxnet model.\n")
        test_img = np.ones((20, 20, 3), dtype=np.uint8)
        test_data = cv2.imencode('.jpg', test_img)[1].tostring()
        prediction = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': test_data})
        prediction = np.squeeze(prediction)
        sys.stdout.write("...length of feature {}    {}.\n".format(len(prediction), "success" * (len(prediction) == 2048)))

    def __create_graph(self):
        # Creates a graph from saved GraphDef file and returns a saver.
        # Creates graph from saved graph_def.pb.
        if not os.path.exists(os.path.join(self.model_dir, 'classify_image_graph_def.pb')):
            self.__download_and_extract_model()
        with tf.gfile.FastGFile(os.path.join(
                self.model_dir, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            _ = tf.import_graph_def(graph_def, name='')

    def __download_and_extract_model(self):
        # Download and extract imgnet tar file.
        self.data_url = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'
        dest_directory = self.model_dir

        if not os.path.exists(dest_directory):
            os.makedirs(dest_directory)
        filename = self.data_url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write("\r>> Downloading %s %.1f%%" % (
                    filename, float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()

            filepath, _ = urllib.request.urlretrieve(self.data_url, filepath, _progress)
            statinfo = os.stat(filepath)
            sys.stdout.write("\nSuccessfully downloaded {} {} bytes.\n".format(filename, statinfo.st_size))
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    def get_feature_from_image(self, img_path):
        """Runs extract the feature from the image.
            Args: img_path: Image file name.

        Returns:  predictions: 2048 * 1 feature vector
        """

        if not tf.gfile.Exists(img_path):
            tf.logging.fatal('File does not exist %s', img_path)
        image_data = tf.gfile.FastGFile(img_path, 'rb').read()

        prediction = self.sess.run(self.softmax_tensor, {'DecodeJpeg/contents:0': image_data})
        prediction = np.squeeze(prediction)
        return prediction


if __name__ == '__main__':
    img_path = RAWDATA + "/sample.jpg"
    "/media/be/DEB_DATA/WORKSPACE/Image-classifier/data/sample.jpg"
    print(img_path)
    feature = ImgNetUtils().get_feature_from_image(img_path=img_path)
    print(len(feature))
