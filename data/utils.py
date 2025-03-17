import os.path as osp

cur_path = osp.dirname(osp.abspath(__file__))

tum_split = osp.join(cur_path, 'tum_split.txt')
tum_split = open(tum_split).read().split()