from src.settings import *


def convert2JPG(folder_path):
    sys.stdout.write("folder: {}\n".format(folder_path))

    tar_ext = ".jpg"

    fns = [fn for fn in os.listdir(folder_path) if os.path.splitext(fn)[1].lower() in IMG_EXTS]
    fns.sort()
    cnt = 0
    for fn in fns:
        sys.stdout.write("\r file: {}".format(fn))
        sys.stdout.flush()

        path = os.path.join(folder_path, fn)
        if not os.path.exists(path):
            sys.stderr.write("no exist {}\n".format(fn))
            continue

        img = cv2.imread(path)
        if img is None:
            sys.stderr.write("unable to read image {}\n".format(fn))
            continue

        new_path = os.path.splitext(path)[0] + tar_ext

        os.remove(path)
        cv2.imwrite(new_path, img)
        cnt += 1

    sys.stdout.write("\n converting is done {}/{}\n".format(cnt, len(fns)))


def unique_id(folder_path, tar_prefix):
    sys.stdout.write("folder: {}\n".format(folder_path))

    fns = [fn for fn in os.listdir(folder_path) if os.path.splitext(fn)[1].lower() in IMG_EXTS]
    fns.sort()
    cnt = 0
    for fn in fns:
        sys.stdout.write("\r file: {}".format(fn))
        sys.stdout.flush()

        if fn[:3] == tar_prefix:
            continue

        path = os.path.join(folder_path, fn)
        if not os.path.exists(path):
            sys.stderr.write("no exist {}\n".format(fn))
            continue

        img = cv2.imread(path)
        if img is None:
            sys.stderr.write("unable to read image {}\n".format(fn))
            continue

        idx = fns.index(fn)
        new_path = os.path.split(path)[0] + "/" + tar_prefix + str(idx) + os.path.splitext(path)[1]

        os.rename(path, new_path)
        cnt += 1

    sys.stdout.write("\n converting is done {}/{}\n".format(cnt, len(fns)))


if __name__ == '__main__':
    # convert2JPG(os.path.join(RAWDATA + "/" + "positive"))
    # convert2JPG(os.path.join(RAWDATA + "/" + "negative"))
    unique_id(os.path.join(RAWDATA + "/" + "positive"), tar_prefix="pos")
    unique_id(os.path.join(RAWDATA + "/" + "negative"), tar_prefix="neg")
