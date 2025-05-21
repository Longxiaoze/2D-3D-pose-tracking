#!/usr/bin/env python3
# detect_2d_line.py

import argparse
import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import cv2
from modeling.afm import AFM
from config import cfg

def parse_args():
    parser = argparse.ArgumentParser(description="detect 2d lines using afm (non-ROS)")
    # 前两个位置参数：输入（文件或文件夹）和输出文件夹
    parser.add_argument('input',  help="输入图像文件或目录")
    parser.add_argument('output', help="输出目录")
    # 第三个位置参数：config 文件
    parser.add_argument('config_file',
                        metavar="CONFIG",
                        help="AFM 配置文件 (yaml)",
                        type=str)
    # 可选 GPU / epoch
    parser.add_argument("--gpu",   type=int, default=0, help="使用的 GPU id")
    parser.add_argument("--epoch", dest="epoch", type=int, default=-1,
                        help="加载哪个 epoch 的权重，默认为最新(-1)")
    # 把剩下的所有参数传给 cfg.merge_from_list
    parser.add_argument("opts",
                        help="额外的 cfg 修改选项，请以 KEY VALUE ... 格式",
                        nargs=argparse.REMAINDER)
    return parser.parse_args()

def detect_and_draw(system, img, config):
    """对单张图像做 AFM 检测并画线，返回 BGR 图像。"""
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    feats = system.detect(img, config)  # N×4
    out = img.copy()
    for i in range(feats.shape[0]):
        line=[round(j) for j in feats[i]]
        cv2.line(out,(line[0],line[1]),(line[2],line[3]),(0,0,255),1,cv2.LINE_AA)

    # cv2.namedWindow('feat_imge', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("feat_imge",(960,540))
    # cv2.imshow('feat_imge', feat_imge)
    # cv2.waitKey(0)
    return out

if __name__ == '__main__':
    args = parse_args()

    # 准备输出目录
    os.makedirs(args.output, exist_ok=True)
    files = []
    if osp.isdir(args.input):
        for fn in sorted(os.listdir(args.input)):
            if fn.lower().endswith(('.png','.jpg','jpeg','bmp')):
                files.append(osp.join(args.input, fn))
    else:
        files = [args.input]

    # 环境变量 & cfg
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg.defrost()
    cfg.SAVE_DIR = ""
    cfg.freeze()
    cfg.merge_from_file(args.config_file)
    if args.opts:
        cfg.merge_from_list(args.opts)

    # 初始化 AFM
    system = AFM(cfg)
    system.model.eval()
    system.load_weight_by_epoch(args.epoch)

    # 遍历处理
    for src in tqdm(files, desc="Detecting"):
        dst = osp.join(args.output, osp.basename(src))
        img = cv2.imread(src)
        if img is None:
            tqdm.write(f"⚠️ 读取失败，跳过 {src}")
            continue

        result = detect_and_draw(system, img, cfg)
        cv2.imwrite(dst, result)
        tqdm.write(f"✅ {osp.basename(src)} -> {dst}")
