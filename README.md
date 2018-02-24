# Mask RCNN
Mask RCNN in TensorFlow

This repo attempts to reproduce this amazing work by Kaiming He et al. : 
[Mask R-CNN](https://arxiv.org/abs/1703.06870)

## Requirements

- [Tensorflow (>= 1.0.0)](https://www.tensorflow.org/install/install_linux)
- [Numpy](https://github.com/numpy/numpy/blob/master/INSTALL.rst.txt)
- [COCO dataset](http://mscoco.org/dataset/#download)
- [Resnet50](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz)

## How-to
1. Go to `./libs/datasets/pycocotools` and run `make`
2. Download [COCO](http://mscoco.org/dataset/#download) dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-records. It takes a while.
3. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`, unzip it, place it into `./data/pretrained_models/`
4. Go to `./libs` and run `make`
5. run `python train/train.py` for training
6. There are certainly some bugs, please report them back, and let's solve them together.

## TODO:
- [x] ROIAlign
- [x] COCO Data Provider
- [x] Resnet50
- [x] Feature Pyramid Network
- [x] Anchor and ROI layer
- [x] Mask layer
- [x] Speedup anchor layer with cython
- [x] Combining all modules together.
- [x] Testing and debugging (in progress)
- [ ] Training / evaluation on COCO
- [ ] Add image summary to show some results
- [ ] Converting ResneXt
- [ ] Training >2 images

## Call for contributions
- Anything helps this repo, including **discussion**, **testing**, **promotion** and of course **your awesome code**.

## Acknowledgment
This repo borrows tons of code from 
- [TFFRCNN](https://github.com/CharlesShang/TFFRCNN)
- [py-faster-rcnn](https://github.com/rbgirshick/py-faster-rcnn) 
- [faster_rcnn](https://github.com/ShaoqingRen/faster_rcnn)
- [tf-models](https://github.com/tensorflow/models)

## License
See [LICENSE](https://github.com/CharlesShang/FastMaskRCNN/blob/master/LICENSE) for details.

