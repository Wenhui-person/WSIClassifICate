import sys
import os
import argparse
import logging
import time
import random

import numpy as np
from multiprocessing import Pool, Value, Lock

import openslide

parser = argparse.ArgumentParser(description='Generate patches from a given '
                                             'list of coordinates')
parser.add_argument('--coords_path',
                    default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/trainset_normal_spot_in_wsi.txt",
                    metavar='COORDS_PATH',
                    type=str, help='Path to the input list of coordinates')
parser.add_argument('--sup_patch_path', default="/home/qianslab/yangwenhui/TUMOR_data_processed/train_patch/sup_patch/normal",
                    metavar='PATCH_PATH', type=str,
                    help='Path to the output directory of patch images')
parser.add_argument('--unsup_patch_path',
                    default="/home/qianslab/yangwenhui/TUMOR_data_processed/train_patch/unsup_patch/", type=str,
                    help="Path to the unlabeled patch directory.")
parser.add_argument('--sampled_split', default=0.15, type=float,
                    help="Sampled.")
parser.add_argument('--patch_size', default=256, type=int, help='patch size, '
                                                                'default 256')
parser.add_argument('--level', default=0, type=int, help='level for WSI, to '
                                                         'generate patches, default 0')
parser.add_argument('--num_process', default=5, type=int,
                    help='number of mutli-process, default 5')

random.seed(111)


def process(opts, args):
    value = 0
    result = []
    for item in opts:
        i, pid, x_center, y_center = item

        x = int(int(x_center) - args.patch_size * 1.5)
        y = int(int(y_center) - args.patch_size * 1.5)
        wsi_path = pid
        sup_path = os.path.join(args.sup_patch_path, str(i) + '.png')
        slide = openslide.OpenSlide(wsi_path)
        for idx in range(3):
            for jdx in range(3):
                img = slide.read_region((x + idx * args.patch_size, y + jdx * args.patch_size),
                                        args.level,
                                        (args.patch_size, args.patch_size)).convert("RGB")
                if idx == jdx == 1:
                    img.save(sup_path)
                else:
                    unsup_path = os.path.join(args.unsup_patch_path,
                                              str(i) + "_" + str(idx) + str(jdx) + '_normal.png')
                    img.save(unsup_path)
                    result.append({"ori_img": unsup_path,
                                   "linked_img": sup_path})

        value += 1
        if (value) % 100 == 0:
            logging.info('{}, {} patches generated...'
                         .format(time.strftime("%Y-%m-%d %H:%M:%S"),
                                 value))

    np.save(os.path.join(args.unsup_patch_path,
                         "../unlabeled_unaug_trainset_data_with" + str(len(result)) + '_img_2.npy'), result)


def run(args):
    logging.basicConfig(level=logging.INFO)

    if not os.path.exists(args.sup_patch_path):
        os.mkdir(args.sup_patch_path)
    if not os.path.exists(args.unsup_patch_path):
        os.mkdir(args.unsup_patch_path)

    # copyfile(args.coords_path, os.path.join(args.patch_path, '../list.txt'))

    opts_list = []
    infile = open(args.coords_path)
    for i, line in enumerate(infile):
        slide_path, x_center, y_center = line.strip('\n').split('\t')
        opts_list.append((i, slide_path, x_center, y_center))
    infile.close()

    ## sample dataset
    sampled_opts_list = random.sample(opts_list,
                                      int(args.sampled_split * len(opts_list)))

    # pool = Pool(processes=args.num_process)
    # pool.map(process, sampled_opts_list)
    process(sampled_opts_list, args)


def main():
    args = parser.parse_args()
    run(args)


if __name__ == '__main__':
    main()
