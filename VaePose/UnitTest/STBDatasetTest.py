# -*- encoding: utf-8 -*-
# @Time : 2019/2/27 16:35

from VaePose.data.STDDataset import getSTBTrainAndTestDataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms

from utils.View import numpyorTensorToPIL, planeHandKeypoint

path = '/Volumes/YF-Code-256/Data/STB/'

# 这里的 ToTensor 会出问题
STBtrans = transforms.Compose([
    transforms.ToTensor(),
])
trainSTBSet, testSTBSet = getSTBTrainAndTestDataset(path, transform=STBtrans)

print(len(trainSTBSet))
trainSTB = DataLoader(trainSTBSet, batch_size=1, shuffle=True)
for data, label in trainSTB:
    print(type(data))
    print(type(label))
    BB_left_i, BB_right_i, SK_color_i, SK_depth = data
    BB, SK = label
    break
print(len(trainSTB))

# im = BB_left_i.numpy()
# im = im.reshape(480, 640, 3)
# BB_left_pil = Image.fromarray(im)
# BB_left_pil.show()

bb_left = numpyorTensorToPIL(BB_left_i)
bb_left.show()
# numpyorTensorToPIL(BB_right_i).show()
# numpyorTensorToPIL(SK_color_i).show()
# numpyorTensorToPIL(SK_depth).show()
planeHandKeypoint(bb_left, BB.numpy()[0])
