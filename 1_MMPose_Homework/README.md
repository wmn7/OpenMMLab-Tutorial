# 步骤

1. 修改配置文件中数据集的位置，是根据当前运行命令的位置。具体来说，修改两个配置文件，将 `data_root` 修改为：
```
data_root = './dataset/'
```
2. 使用 MMDetection 训练检测模型，结果会保存在 work_dirs 里面；
```
nohup python mmdet_train.py ./config/rtmdet_tiny_ear.py > mmdet_output.log &
```
3. 使用 MMPose 训练关键点模型，同样结果会保存在 work_dirs 里面；
```
nohup python mmpose_train.py ./config/rtmpose-s-ear.py > mmpose_output.log &
```
4. 模型权重精简转换，去掉模型中训练部分的信息。
```
python publish_model.py work_dirs/rtmdet_tiny_ear/epoch_200.pth checkpoint/rtmdet_tiny_ear.pth
python publish_model.py work_dirs/rtmpose-s-ear/epoch_300.pth checkpoint/rtmpose-s-ear.pth
```
5. 使用模型进行预测。
```
python topdown_demo_with_mmdet.py \
        config/rtmdet_tiny_ear.py \
        checkpoint/rtmdet_tiny_ear.pth \
        config/rtmpose-s-ear.py \
        checkpoint/rtmpose-s-ear.pth \
        --input dataset/images/DSC_5387.jpg \
        --output-root dataset/ \
        --device cuda:0 \
        --bbox-thr 0.5 \
        --kpt-thr 0.5 \
        --nms-thr 0.3 \
        --radius 36 \
        --thickness 30 \
        --draw-bbox \
        --draw-heatmap \
        --show-kpt-idx
```

# 结果

详细的训练日志可以查看 `mmdet_output.log` 和 `mmpose_output.log` 文件。det 和 pose 的最终结果分别是：

mmdet:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.817
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.970
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.817
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.845
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.845
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.845
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.845
```

mmpose:

```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.739
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.944
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.739
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 20 ] =  0.788
 Average Recall     (AR) @[ IoU=0.50      | area=   all | maxDets= 20 ] =  1.000
 Average Recall     (AR) @[ IoU=0.75      | area=   all | maxDets= 20 ] =  0.952
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets= 20 ] = -1.000
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets= 20 ] =  0.788
```

最终的关键点预测结果如下：

![exp_result](./dataset/DSC_5387.jpg)