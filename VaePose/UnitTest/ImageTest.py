# -*- encoding: utf-8 -*-
# @Time : 2019/2/27 16:10

import os
import numpy as np
from  PIL import Image

# Folder_path = 'D:/Dataset/Hand Pose 2d to 3d/STB/B1Random/'
Folder_path = '/Volumes/YF-Code-256/Data/STB/B1Counting/'
file_name = 'SK_depth_0.png'
image = Image.open(os.path.join(Folder_path, file_name)).convert('RGB')
print(type(image))
print((image.size))
# image.show()
RGB = np.array(image)
print(RGB.shape)
print(RGB[240][320])
RGB[:,:,1] = 0
im = Image.fromarray(RGB)
im.show()
Depth = RGB[:,:,1] * 256 + RGB[:,:,0]
de = Image.fromarray(np.uint8(Depth / np.max(Depth) *255))
de.show()