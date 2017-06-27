import numpy as np
import tensorflow as tf
from PIL import Image, ImageFont, ImageDraw, ImageEnhance

FLAGS = tf.app.flags.FLAGS
_DEBUG = False

def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    #print("image")
    #print(image)
    #norm_image = np.uint8(image/np.max(np.abs(image))*255.0)
    norm_image = np.uint8(image/0.1*127.0 + 127.0)
    #print("norm_image")
    #print(norm_image)
    source_img = Image.fromarray(norm_image)
    return source_img.save(FLAGS.train_dir + 'test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def draw_bbox(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, prob=None):
    #print(prob[:,label])
    source_img = Image.fromarray(image)
    b, g, r = source_img.split()
    source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    color = '#0000ff'
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None:
                if prob is not None:
                    if (prob[i,label[i]] > 0.5) and (label[i] > 0):
                        if gt_label is not None:
                            text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i])
                            if label[i] != gt_label[i]:
                                color = '#ff0000'#draw.text((2+bbox[i,0], 2+bbox[i,1]), cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]), fill='#ff0000')
                            else:
                                color = '#0000ff'  
                        else: 
                            text = cat_id_to_cls_name(label[i])
                        draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                        if _DEBUG is True:
                            print("plot",label[i], prob[i,label[i]])
                        draw.rectangle(box,fill=None,outline=color)
                    else: 
                        if _DEBUG is True:
                            print("skip",label[i], prob[i,label[i]])
                else:
                    text = cat_id_to_cls_name(label[i])
                    draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                    draw.rectangle(box,fill=None,outline=color)


    return source_img.save(FLAGS.train_dir + '/est_imgs/test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def cat_id_to_cls_name(catId):
    cls_name = np.array([  'background', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                       'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
                       'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                       'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
                       'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
                       'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                       'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
                       'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
                       'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
                       'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
                       'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
                       'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven',
                       'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
                       'scissors', 'teddy bear', 'hair drier', 'toothbrush'])
    return cls_name[catId]

    # real_name = np.array([  "background",
    #                         "person",
    #                         "bicycle",
    #                         "car",
    #                         "motorcycle",
    #                         "airplane",
    #                         "bus",
    #                         "train",
    #                         "truck",
    #                         "boat",
    #                         "traffic light",
    #                         "fire hydrant",
    #                         "stop sign",
    #                         "parking meter",
    #                         "bench",
    #                         "bird",
    #                         "cat",
    #                         "dog",
    #                         "horse",
    #                         "sheep",
    #                         "cow",
    #                         "elephant",
    #                         "bear",
    #                         "zebra",
    #                         "giraffe",
    #                         "backpack",
    #                         "umbrella",
    #                         "handbag",
    #                         "tie",
    #                         "suitcase",
    #                         "frisbee",
    #                         "skis",
    #                         "snowboard",
    #                         "sports ball",
    #                         "kite",
    #                         "baseball bat",
    #                         "baseball glove",
    #                         "skateboard",
    #                         "surfboard",
    #                         "tennis racket",
    #                         "bottle",
    #                         "wine glass",
    #                         "cup",
    #                         "fork",
    #                         "knife",
    #                         "spoon",
    #                         "bowl",
    #                         "banana",
    #                         "apple",
    #                         "sandwich",
    #                         "orange",
    #                         "broccoli",
    #                         "carrot",
    #                         "hot dog",
    #                         "pizza",
    #                         "donut",
    #                         "cake",
    #                         "chair",
    #                         "couch",
    #                         "potted plant",
    #                         "bed",
    #                         "dining table",
    #                         "toilet",
    #                         "tv",
    #                         "laptop",
    #                         "mouse",
    #                         "remote",
    #                         "keyboard",
    #                         "cell phone",
    #                         "microwave",
    #                         "oven",
    #                         "toaster",
    #                         "sink",
    #                         "refrigerator",
    #                         "book",
    #                         "clock",
    #                         "vase",
    #                         "scissors",
    #                         "teddy bear",
    #                         "hair drier",
    #                         "toothbrush",])

        # cls_name = np.array(["background",
        #                     "person",
        #                     "bicycle",
        #                     "car",
        #                     "motorcycle",
        #                     "airplane",
        #                     "bus",
        #                     "train",
        #                     "truck",
        #                     "boat",
        #                     "traffic_light",
        #                     "fire_hydrant",
        #                     "street_sign",
        #                     "stop_sign",
        #                     "parking_meter",
        #                     "bird",
        #                     "cat",
        #                     "bench",
        #                     "horse",
        #                     "dog",
        #                     "sheep",
        #                     "cow",
        #                     "bear",
        #                     "elephant",
        #                     "giraffe",
        #                     "zebra",
        #                     "hat",
        #                     "backpack",
        #                     "snowboard",
        #                     "shoe",
        #                     "eye_glasses",
        #                     "hand_bag",
        #                     "umbrella",
        #                     "suitcase",
        #                     "frisbee",
        #                     "skis",
        #                     "tie",
        #                     "sports_ball",
        #                     "surfboard",
        #                     "tennis_racket",
        #                     "baseball_glove",
        #                     "skateboard",
        #                     "kite",
        #                     "fork",
        #                     "bottle",
        #                     "plate",
        #                     "wine_glass",
        #                     "cup",
        #                     "baseball_bat",
        #                     "knife",
        #                     "spoon",
        #                     "broccoli",
        #                     "carrot",
        #                     "apple",
        #                     "sandwich",
        #                     "orange",
        #                     "bowl",
        #                     "banana",
        #                     "hot_dog",
        #                     "pizza",
        #                     "bed",
        #                     "dining_table",
        #                     "chair",
        #                     "couch",
        #                     "potted_plant",
        #                     "donut",
        #                     "mirror",
        #                     "cake",
        #                     "window",
        #                     "desk",
        #                     "toilet",
        #                     "door",
        #                     "tv",
        #                     "laptop",
        #                     "book",
        #                     "remote",
        #                     "vase",
        #                     "cell_phone",
        #                     "microwave",
        #                     "oven",
        #                     "toaster",
        #                     "sink",
        #                     "refrigerator",
        #                     "blender",
        #                     "mouse",
        #                     "clock",
        #                     "keyboard",
        #                     "scissors",
        #                     "teddy_bear",
        #                     "hair_dryer",
        #                     "toothbrush",
        #                     "hair_brush"])
