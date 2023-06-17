# ControlNet 毛坯房2效果图


直接运行下面的命令生成效果图，其中 [config](./config.py) 是 ControlNet Canny 的配置文件，包含模型的结构：

```shell
python controlnet_canny.py
```

原始的**毛坯房**如下图所示：

![](./test_canny.jpeg)

边缘检测的结果如下所示：

![](./control_0.png)

最终的效果图如下所示：

![](./sample_0.png)