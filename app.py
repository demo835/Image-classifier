import os
import cv2
import sys
import base64
from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify, send_file
from werkzeug.utils import secure_filename

from test import proc

app = Flask(__name__)

# predefined the location for saving the uploaded files
UPLOAD_DIR = 'data/'


def allowed_file(filename):
    return '.' in filename and os.path.splitext(filename)[1].lower() in [".jpg", ".jpeg", ".png"]


@app.route('/')
def detect():
    """
        initial rendering of the web interface
    """
    return render_template('detect.html')


@app.route('/submit', methods=['POST'])
def submit():

    if len(request.files) > 0:
        file = request.files['file']
        doc_fn = secure_filename(file.filename)

        sys.stdout.write("\n>>> uploading image\n")
        sys.stdout.write("\t{}\n".format(request))
        print(request)

        if file and allowed_file(file.filename):
            try:
                # check its directory for uploading the requested file
                if not os.path.isdir(UPLOAD_DIR):
                    os.mkdir(UPLOAD_DIR)

                # remove all the previous processed document file
                for fname in os.listdir(UPLOAD_DIR):
                    path = os.path.join(UPLOAD_DIR, fname)
                    if os.path.isfile(path):
                        os.remove(path)

                # save the uploaded document on UPLOAD_DIR
                file.save(os.path.join(UPLOAD_DIR, doc_fn))
                sys.stdout.write("\tsave image {}\n".format(doc_fn))

                # ocr progress with the uploaded files
                sys.stdout.write("\tsuccessfully uploaded.\n")
                dst_file_path = os.path.join(UPLOAD_DIR, doc_fn)

                #
                #
                ret = proc(img_path=dst_file_path)
                #
                #

                if ret:
                    sys.stdout.write("\n successfully finished\n")
                    return send_file(dst_file_path, mimetype='image/jpeg')
                else:
                    sys.stdout.write("\n some error\n")
                    return "Error"

            except Exception as e:
                str_msg = '\tException: {}'.format(e)
                sys.stdout.write("\t exception :{}\n".format(str_msg))
                return str_msg

        else:
            str_msg = "\tnot allowed file format {}.\n".format(doc_fn)
            sys.stdout.write(str_msg)
            return str_msg


if __name__ == '__main__':
    # open the port 5000 to connect betweeen client and server
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000)),
        debug=False,
        threaded=True,
    )
