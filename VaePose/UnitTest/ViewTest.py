# -*- encoding: utf-8 -*-
# @Time : 2019/3/8 13:48

from VaePose.utils import View
from VaePose.UnitTest.RHDLabelTest import getRHDLabel
import numpy as np


def view_hand_3d_test():
    def move_wrist(middle_finger_coords, wrist_coords, shift_factor):
        d = np.sqrt(np.sum((middle_finger_coords - wrist_coords) ** 2))
        vec = middle_finger_coords - wrist_coords
        vec /= np.linalg.norm(vec)
        wrist_coords += vec * d * shift_factor
        return wrist_coords
    ann_all = getRHDLabel()

    for sample_id, anno in ann_all.items():  # len(anno_all) == 41258
        print(sample_id)
        left_right_hand_idx = [(0, 21), (21, 42)]
        for lr_idx in left_right_hand_idx:
            fr, to = lr_idx
            if anno['uv_vis'][fr:to, 2].sum() == 21:
                K = anno['K']
                d3d = anno['xyz'][fr:to]
                # View.view_hand_3d(d3d,flag=str(sample_id))
                middle_finger_offset = 3 * 4
                d3d[0, :] = move_wrist(
                    d3d[middle_finger_offset, :],
                    d3d[0, :],
                    shift_factor=0.5)
                d2d = anno['uv_vis'][fr:to, :2]
                # View.view_hand_2d(d2d, show=True)
                projected = np.matmul(d3d[0, :], np.transpose(K))  # projected 也是一个三维的
                # print(d2d)
                # print(d2d - projected[:2] / projected[2])
                d2d[0, :] = projected[:2] / projected[2]  # 前面两维，除第三维, 这里只修改了第一维。
                # 第一根手指的长度。0.0346
                len_first_index_bone = np.sqrt(np.sum((d3d[12, :] - d3d[11, :]) ** 2))
                View.view_hand_2d(d2d, show=False,flag=str(sample_id))
                _, ed3d = View.view_hand_3d(d3d, show=False,flag=str(sample_id), echart=True)
                print(len_first_index_bone)
                print(ed3d)

if __name__ == '__main__':

    view_hand_3d_test()