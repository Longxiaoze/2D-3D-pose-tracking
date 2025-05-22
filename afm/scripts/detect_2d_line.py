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
    parser = argparse.ArgumentParser(description="Detect 2D lines using AFM (non-ROS)")
    # First two positional arguments: input (file or directory) and output directory
    parser.add_argument('input',  help="Input image file or directory")
    parser.add_argument('output', help="Output directory")
    # Third positional argument: configuration file
    parser.add_argument('config_file',
                        metavar="CONFIG",
                        help="AFM configuration file (YAML)",
                        type=str)
    # Optional GPU ID and epoch number
    parser.add_argument("--gpu",   type=int, default=0, help="GPU ID to use")
    parser.add_argument("--epoch", dest="epoch", type=int, default=-1,
                        help="Which epoch weights to load (default: latest, -1)")
    # Remaining args passed to cfg.merge_from_list
    parser.add_argument("opts",
                        help="Additional config options in KEY VALUE ... format",
                        nargs=argparse.REMAINDER)
    return parser.parse_args()

def detect_and_draw(system, img, config):
    """Run AFM detection on a single image and draw lines; returns a BGR image."""
    # If grayscale, convert to BGR
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    # Detect line features (N×4 array)
    feats = system.detect(img, config)
    out = img.copy()
    # Draw each detected line in red
    for i in range(feats.shape[0]):
        line = [round(j) for j in feats[i]]
        cv2.line(out, (line[0], line[1]), (line[2], line[3]), (0, 0, 255), 1, cv2.LINE_AA)

    # # Optional display code (disabled by default)
    # cv2.namedWindow('feat_image', cv2.WINDOW_NORMAL)
    # cv2.resizeWindow("feat_image", (960, 540))
    # cv2.imshow('feat_image', out)
    # cv2.waitKey(0)

    return out

if __name__ == '__main__':
    args = parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output, exist_ok=True)
    files = []
    # If input is a directory, gather all supported image files
    if osp.isdir(args.input):
        for fn in sorted(os.listdir(args.input)):
            if fn.lower().endswith(('.png', '.jpg', 'jpeg', '.bmp')):
                files.append(osp.join(args.input, fn))
    else:
        files = [args.input]

    # Set GPU device and load configuration
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    cfg.merge_from_file(args.config_file)
    # if args.opts:
    #     cfg.merge_from_list(args.opts)

    # Initialize AFM system
    system = AFM(cfg)
    system.model.eval()
    system.load_weight_by_epoch(args.epoch)

    # Process each file
    for src in tqdm(files, desc="Detecting"):
        dst = osp.join(args.output, osp.basename(src))
        img = cv2.imread(src)
        if img is None:
            tqdm.write(f"⚠️  Failed to read, skipping {src}")
            continue

        result = detect_and_draw(system, img, cfg)
        cv2.imwrite(dst, result)
        tqdm.write(f"✅  {osp.basename(src)} -> {dst}")
