# FSAF
This is an implementation of [FSAF](https://arxiv.org/abs/1903.00621) on keras and Tensorflow. The project is based on [qqwweee/keras-yolo3](https://github.com/qqwweee/keras-yolo3) and [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet). 
Thanks for their hard work. 

As the authors write, FASF module can be plugged into any single-shot detectors with FPN-like structure smoothly. 
I have also tried on yolo3. Anchor-free yolo3(with FSAF) gets a comparable performance with the anchor-based counterpart. But you don't need to pre-compute the anchor sizes any more.
And it is much better and faster than the one based on retinanet.
  
It can also converge quite quickly. After the first epoch(batch_size=64, steps=1000), we can get an mAP<sub>50</sub> of 0.6xxx on val dataset.

## Test
1. I trained on Pascal VOC2012 trainval.txt + Pascal VOC2007 train.txt, and validated on Pascal VOC2007 val.txt. There are 14041 images for training and 2510 images for validation.
2. The best evaluation result (score_threshold=0.01, mAP<sub>50</sub>, image_size=416) on VOC2007 test is 0.8358. I have only trained once.
3. Pretrained yolo and fsaf weights are here. [baidu netdisk](https://pan.baidu.com/s/1QoGXnajcohj9P4yCVwJ4Yw), extract code: qab7
4. `python3 yolo/inference.py` to test your image by specifying image path and model path there. 

## Train
### build dataset (Pascal VOC, other types please refer to [fizyr/keras-retinanet](https://github.com/fizyr/keras-retinanet))
* Download VOC2007 and VOC2012, copy all image files from VOC2007 to VOC2012.
* Append VOC2007 train.txt to VOC2012 trainval.txt.
* Overwrite VOC2012 val.txt by VOC2007 val.txt.
### train
* **STEP1**: `python3 yolo/train.py --freeze-body darknet --gpu 0 --batch-size 32 --random-transform pascal datasets/VOC2012` to start training with lr=1e-3 then set lr=1e-4 when val mAP continue to drop.
* **STEP2**: `python3 yolo/train.py --snapshot <xxx> --freeze-body none --gpu 0 --batch-size 32 --random-transform pascal datasets/VOC2012` to start training with lr=1e-5 and then set lr=1e-6 when val mAP contines to drop.
## Evaluate
* `python3 yolo/eval/common.py` to evaluate by specifying model path there.
