import numpy as np
import libs.configs.config_v1 as cfg
from PIL import Image, ImageFont, ImageDraw, ImageEnhance
from scipy.misc import imresize

FLAGS = cfg.FLAGS
_DEBUG = False

def draw_img(step, image, name='', image_height=1, image_width=1, rois=None):
    img = np.uint8(image/0.1*127.0 + 127.0)
    img = Image.fromarray(img)
    return img.save(FLAGS.train_dir + 'test_' + name + '_' +  str(step) +'.jpg', 'JPEG')

def draw_bbox(step, image, name='', image_height=1, image_width=1, bbox=None, label=None, gt_label=None, mask=None, prob=None, iou=None, vis_th=0.5, vis_all=False, ignore_bg=True):
    source_img = Image.fromarray(image)
    b, g, r = source_img.split()
    source_img = Image.merge("RGB", (r, g, b))
    draw = ImageDraw.Draw(source_img)
    color = '#0000ff'
    if mask is not None:
        m = np.array(mask*255.0)
        m = np.transpose(m,(0,3,1,2))
    if bbox is not None:
        for i, box in enumerate(bbox):
            if label is not None and not np.all(box==0):
                if prob is not None:
                    if ((prob[i,label[i]] > vis_th) or (vis_all is True)) and ((ignore_bg is True) and (label[i] > 0)) :
                        if gt_label is not None:
                            if gt_label is not None and len(iou) > 1:
                                text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]) + ' : ' + str(iou[i])[:3]
                            else:
                                text  = cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]) + ' : ' + str(prob[i][label[i]])[:4]
                            
                            if label[i] != gt_label[i]:
                                color = '#ff0000'#draw.text((2+bbox[i,0], 2+bbox[i,1]), cat_id_to_cls_name(label[i]) + ' : ' + cat_id_to_cls_name(gt_label[i]), fill='#ff0000')
                            else:
                                color = '#0000ff'  
                        else: 
                            text = cat_id_to_cls_name(label[i]) + ' : ' +  "{:.3f}".format(prob[i][label[i]])  #str(i)#+
                        draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)

                        if _DEBUG is True:
                            print("plot",label[i], prob[i,label[i]])
                        draw.rectangle(box,fill=None,outline=color)

                        if mask is not None:
                            # print("mask number: ",i)
                            box = np.floor(box).astype('uint16')
                            bbox_w = box[2]-box[0]
                            bbox_h = box[3]-box[1]
                            mask_color_id = np.random.randint(35)
                            color_img = color_id_to_color_code(mask_color_id)* np.ones((bbox_h,bbox_w,1)) * 255
                            color_img = Image.fromarray(color_img.astype('uint8')).convert('RGBA')
                            #color_img = Image.new("RGBA", (bbox_w,bbox_h), np.random.rand(1,3) * 255 )
                            resized_m = imresize(m[i][label[i]], [bbox_h, bbox_w], interp='bilinear') #label[i]
                            resized_m[resized_m >= 128] = 128
                            resized_m[resized_m < 128] = 0
                            resized_m = Image.fromarray(resized_m.astype('uint8'), 'L')
                            #print(box)
                            #print(resized_m)
                            
                            source_img.paste(color_img , (box[0], box[1]), mask=resized_m)

                        #return source_img.save(FLAGS.train_dir + 'est_imgs/' + name + '_' +  str(step) +'.jpg', 'JPEG')
                    
                    else: 
                        if _DEBUG is True:
                            print("skip",label[i], prob[i,label[i]])
                else:
                    text = cat_id_to_cls_name(label[i])
                    draw.text((2+bbox[i,0], 2+bbox[i,1]), text, fill=color)
                    draw.rectangle(box,fill=None,outline=color)

    return source_img.save(FLAGS.train_dir + 'est_imgs/' + name + '_' +  str(step) +'.jpg', 'JPEG')

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

def color_id_to_color_code(colorId):
    color_code = np.array([[178, 31, 53],
                           [216, 39, 53],
                           [255, 116, 53],
                           [255, 161, 53],
                           [255, 203, 53],
                           [255, 255, 53],
                           [0, 117, 58],
                           [0, 158, 71],
                           [22, 221, 53],
                           [0, 82, 165],
                           [0, 121, 231],
                           [0, 169, 252],
                           [104, 30, 126],
                           [125, 60, 181],
                           [189, 122, 246],
                           [234, 62, 112],
                           [198, 44, 58],
                           [243, 114, 82],
                           [255, 130, 1],
                           [255, 211, 92],
                           [138, 151, 71],
                           [2, 181, 160],
                           [75, 196, 213],
                           [149, 69, 103],
                           [125, 9, 150],
                           [169, 27, 176],
                           [198, 30, 153],
                           [207, 0, 99],
                           [230, 21, 119],
                           [243, 77, 154],
                           [144, 33, 71],
                           [223, 40, 35],
                           [247, 106, 4],
                           [206, 156, 72],
                           [250, 194, 0],
                           [254, 221, 39],
                           ])
    return color_code[colorId]
