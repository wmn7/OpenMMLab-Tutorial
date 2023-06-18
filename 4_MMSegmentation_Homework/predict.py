import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from mmengine import Config
from mmengine.runner import Runner
from mmseg.utils import register_all_modules
from mmseg.apis import init_model, inference_model, show_result_pyplot


import mmcv
import cv2

if __name__ == '__main__':
    # 载入 config 配置文件
    cfg = Config.fromfile('./pspnet_config.py')

    # register all modules in mmseg into the registries
    # do not init the default scope here because it will be init in the runner
    register_all_modules(init_default_scope=False)
    runner = Runner.from_cfg(cfg)

    checkpoint_path = './work_dirs/iter_3000.pth'
    model = init_model(cfg, checkpoint_path, 'cuda:0')
    img_path = './watermelon_dataset/img_dir/val/5b3c8018N634d43bd.jpg'
    img = mmcv.imread(img_path)
    result = inference_model(model, img)

    print('Start Visualization.')
    visualization = show_result_pyplot(model, img, result, opacity=0.7, show=False, out_file='./pred.jpg')

    # vis with legend
    # 获取类别名和调色板
    classes = ['background', 'red', 'green', 'white', 'seedBlack', 'seedWhite']
    palette = [[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]
    opacity = 0.15 # 透明度，越大越接近原图

    # 将分割图按调色板染色
    pred_mask = result.pred_sem_seg.data[0].cpu().numpy()
    seg_map = pred_mask.astype('uint8')
    seg_img = Image.fromarray(seg_map).convert('P')
    seg_img.putpalette(np.array(palette, dtype=np.uint8))

    plt.figure(figsize=(10, 7))
    im = plt.imshow(((np.array(seg_img.convert('RGB')))*(1-opacity) + img*opacity) / 255)

    # 为每一种颜色创建一个图例
    patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(6)]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig('./predict.png')