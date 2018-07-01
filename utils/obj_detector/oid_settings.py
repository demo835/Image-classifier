
from src.settings import ROOT, os

cur_dir = ROOT + "/utils/obj_detector"

OID_MODEL = cur_dir + "/model"
OID_LABEL = OID_MODEL + "/oid_label_v4"

TARGET_OBJ_IDS = [100, 263, 326, 381, 403]  # fishes

CLR_RECT = (0, 0, 255)
CLR_TXT = (0, 0, 255)
