# Fast Mask RCNN
Mask RCNN in TensorFlow  
This repo attempts to reporduce this amazing work by Kaiming He.
[Mask RCNN](https://arxiv.org/abs/1703.06870).

## Known Problem
- [ ] Allocate-Zero-Memory Error.
  Invoking sess.run([update_op, total_loss]) outputs error mssg, like,
  ```
  E tensorflow/core/common_runtime/bfc_allocator.cc:244] tried to allocate 0 bytes
  W tensorflow/core/common_runtime/allocator_retry.cc:32] Request to allocate 0 bytes
  F tensorflow/core/common_runtime/gpu/gpu_device.cc:104] EigenAllocator for GPU ran out of memory when allocating 0. See error logs for more detailed info.
  ```
  However, Invoking sess.run(update_op_) won't output any errors. 
  I've spent 2 days on this, looked into tf community, found nothing..
  We can't see any outputs, neither total_loss or summaries, we're totally blind.

- [ ] 

## How-to
1. Download coco dataset, place it into `./data`, then run `python download_and_convert_data.py` to build tf-record. It takes a while.
2. Download pretrained resnet50 model, `wget http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz`, unzip it, place it into `./data/pretrained_models/`
3. run `python test/resnet50_test.py` for training 
4. There are certainly some bugs, please report them back, and let's solve them togather.
## Timeline
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
- [ ] Data queue
- [ ] Data agument
- [ ] Other backbone networks
- [ ] Training >2 images

## Call for contribution
- Anything helps this repo, including **discussion**, **testing**, **promotion** and of course **your awesome code**. 
