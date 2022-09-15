# -*- Coding: utf-8 -*-
import os
import argparse
import logging

import numpy as np
from utils import mask, sampled, dataset_cfg

traindata_tumor = dataset_cfg.TRAINSET_TUMOR
traindata_normal = dataset_cfg.TRAINSET_NORMAL
test_tumor = dataset_cfg.TEST_TUMOR
test_normal = dataset_cfg.TEST_NORMAL


def sampled_tumor(args, normal, tumor, *trainset):
    root = os.path.join(args.wsi_path, "tumor")
    root_xml = os.path.join(args.wsi_path, "xml")

    f = open(normal, "a+")
    f2 = open(tumor, "a+")
    for slide_name in trainset:
        slide_path = os.path.join(root, slide_name)
        xml_path = os.path.join(root_xml, slide_name.rsplit('.')[0] + '.xml')
        # slide = openslide.OpenSlide(slide_path)
        # print(slide_name)
        print(slide_path)
        maskoj = mask.Mask(slide_path, 50, 2)
        tumor_mask = maskoj.tumor_mask(xml_path)
        norml_mask = maskoj.normal_mask(xml_path)
        slide = maskoj.slide
        # print(slide.level_dimensions[0])
        factor = slide.level_downsamples[2]
        # print(factor)
        # print(tumor_mask.shape)

        spots_tumor = sampled.random_sampled(tumor_mask, 1000)
        spots_normal = sampled.random_sampled(norml_mask, 600)
        spots_tumor = np.round(spots_tumor * factor).astype(np.int32)
        spots_normal = np.round(spots_normal * factor).astype(np.int32)
        # print(spots_tumor)
        for it in spots_normal:
            f.write(slide_path + '\t')
            f.write(str(it[0]) + '\t')
            f.write(str(it[1]) + "\n")
        for it in spots_tumor:
            f2.write(slide_path + '\t')
            f2.write(str(it[0]) + "\t")
            f2.write(str(it[1]) + "\n")

    f2.close()
    f.close()


def sampled_normal(args, normal, *trainset):
    root = os.path.join(args.wsi_path, "normal")

    f = open(normal, "a+")
    for slide_name in trainset:
        slide_path = os.path.join(root, slide_name)
        print(slide_path)
        maskoj = mask.Mask(slide_path, 50, 2)
        tussion_mask = maskoj.tissue_mask()
        slide = maskoj.slide
        factor = slide.level_downsamples[2]

        spot_normal = sampled.random_sampled(tussion_mask, 400)
        spot_normal = np.round(spot_normal * factor).astype(np.int32)
        for it in spot_normal:
            f.write(slide_path + '\t')
            f.write(str(it[0]) + '\t')
            f.write(str(it[1]) + "\n")

    f.close()


def main(args):
    sampled_train_tumor = os.path.join(args.output, "trainset_tumor_spot_in_wsi.txt")
    sampled_train_normal = os.path.join(args.output, "trainset_normal_spot_in_wsi.txt")
    if os.path.exists(sampled_train_normal):
        os.remove(sampled_train_normal)
    if os.path.exists(sampled_train_tumor):
        os.remove(sampled_train_tumor)
    sampled_tumor(args, sampled_train_normal, sampled_train_tumor, *traindata_tumor)
    sampled_normal(args, sampled_train_normal, *traindata_normal)

    sampled_test_tumor = os.path.join(args.output, "testset_tumor_spot_in_wsi.txt")
    sampled_test_normal = os.path.join(args.output, "testset_normal_spot_in_wsi.txt")
    if os.path.exists(sampled_test_tumor):
        os.remove(sampled_test_tumor)
    if os.path.exists(sampled_test_normal):
        os.remove(sampled_test_normal)
    sampled_tumor(args, sampled_test_normal, sampled_test_tumor, *test_tumor)
    sampled_normal(args, sampled_test_normal, *test_normal)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Patch extract for training tile classification model.")
    parser.add_argument('--wsi_path', default="/media/qianslab/Data/Cervical-cancer-data/wsi/", type=str,
                        help="The path to whole slide image folder.")
    parser.add_argument('--output',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files",
                        type=str,
                        help="The path to output folder.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)
