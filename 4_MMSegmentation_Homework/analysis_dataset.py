"""
- 共有 5 个类别的数据
类别名称	类别语义	标注类别	灰度图像素值
/	背景	/	0
red	西瓜红瓤	多段线（polygon）	1
green	西瓜外壳	多段线（polygon）	2
white	西瓜白皮	多段线（polygon）	3
seed-black	西瓜黑籽	多段线（polygon）	4
seed-white	西瓜白籽	多段线（polygon）	5
"""
import cv2
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

if __name__ == '__main__':
    mask_path = './watermelon_dataset/ann_dir/train/04_35-2.png'
    mask = cv2.imread(mask_path)
    print(np.unique(mask)) # 打印包含多少种类别
    # [0 1 2 3 4 5]

    # 可视化 mask
    classes = ['background', 'red', 'green', 'white', 'seedBlack', 'seedWhite']
    palette = [[132,41,246], [228,193,110], [152,16,60], [58,221,254], [41,169,226], [155,155,155]]

    mask = mask[:,:,0]
    mask = Image.fromarray(mask).convert('P')
    mask.putpalette(np.array(palette, dtype=np.uint8))
    plt.imshow(mask) # 只取单通道
    plt.axis('off')
    # 为每一种颜色创建一个图例
    patches = [mpatches.Patch(color=np.array(palette[i])/255., label=classes[i]) for i in range(len(classes))]
    plt.legend(handles=patches, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0., fontsize='large')
    plt.tight_layout()

    plt.savefig('./test_vis_mask.png')