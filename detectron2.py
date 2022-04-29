import cv2

import detectron2
from detectron2.utils.logger import setup_logger
setup_logger()

from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo

print("import calisti")

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.7
cfg.MODEL.DEVICE = "cuda"         

print("config calisti")

img = cv2.imread("/truba_scratch/agasi/coco_dataset/14.jpg",cv2.IMREAD_UNCHANGED)
width=224
height=224
dim = (width, height)
img = cv2.resize(img,dim,interpolation = cv2.INTER_AREA)

print("imread calisti")

predictor = DefaultPredictor(cfg)
outputs = predictor(img)

print("predictor calisti")

print(outputs["instances"].pred_classes)
print(outputs["instances"].pred_boxes)
print(outputs["instances"].scores)

print("bitti")