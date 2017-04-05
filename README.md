# Fast Mask RCNN
Mask RCNN in TensorFlow 

## See you in one week
This repo attempts to reporduce this amazing work by Kaiming He.
[Mask RCNN](https://arxiv.org/abs/1703.06870).
The original work involves two stages, a pyramid Faster-RCNN for object detection and another network (with the same structure) for instance level segmentation. 
However, we aim to build an end-to-end framework in the first place. 

## Timeline
- [x] ROIAlign
- [x] COCO Data Provider
- [x] Resnet50
- [x] Feature Pyramid Network
- [x] Anchor and ROI layer
- [x] Mask layer
- [x] Speedup anchor layer with cython
- [x] Combining all modules together. 
- [ ] Testing and debugging (in progress)
- [ ] Training / evaluation on COCO
- [ ] Data queue
- [ ] Data agument
- [ ] Other backbone networks
- [ ] Training >2 images

## Call for contribution
- Anything helps this repo, including **discussion**, **testing**, **promotion** and of course **your awesome code**. 