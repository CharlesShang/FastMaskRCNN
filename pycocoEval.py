
import matplotlib.pyplot as plt
# from data.coco.PythonAPI.pycocotools.coco import COCO
# from data.coco.PythonAPI.pycocotools.cocoeval import COCOeval
from libs.datasets.pycocotools.coco import COCO
from libs.datasets.pycocotools.cocoeval import COCOeval
import numpy as np
import skimage.io as io
import pylab
import json

pylab.rcParams['figure.figsize'] = (10.0, 8.0)

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)

#initialize COCO ground truth api
dataDir='data/coco/'
dataType='train2014'#val2014
annFile = '%s/annotations/%s_%s.json'%(dataDir,prefix,dataType)
cocoGt=COCO(annFile)


#initialize COCO detections api
# resFile='%s/results/%s_%s_fake%s100_results.json'
# resFile = resFile%(dataDir, prefix, dataType, annType)
resFile = 'output/mask_rcnn/results.json'
cocoDt=cocoGt.loadRes(resFile)

with open(resFile) as results:
	res = json.load(results)

imgIds = []

for inst in res:
	imgIds.append(inst['image_id'])

# imgIds=[378962, 116819, 378967, 378968, 116825]

# running evaluation
cocoEval = COCOeval(cocoGt,cocoDt,annType)
cocoEval.params.imgIds  = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()