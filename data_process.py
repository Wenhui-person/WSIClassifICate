import os
import argparse
import logging
import sys

import numpy as np
from PIL import Image

from data.autoaugment import ImageNetPolicy

TEST_RAW_DATA = 'test_raw_data'
UNLABELED_RAW_DATA = 'unlabeled_raw_data'

Image.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = None


def load_dataset(args):
    if args.data_type == 'sup':
        data = []
        train_dataset_dir = args.raw_data_path
        if sys.version_info >= (3, 5):
            # Faster and available in python 3.5 and above
            classes = [d.name for d in os.scandir(train_dataset_dir) if d.is_dir()]
        else:
            classes = [d for d in os.listdir(train_dataset_dir) if os.path.isdir(os.path.join(train_dataset_dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}

        for dir in os.listdir(train_dataset_dir):
            dir_path = os.path.join(train_dataset_dir, dir)
            if os.path.isdir(dir_path):
                print(dir_path)
                for file in os.listdir(dir_path):
                    data.append({"img": os.path.join(dir_path,file),
                                 "label": class_to_idx[dir]})
        return data, class_to_idx

    elif args.data_type == 'unsup':
        # data = []
        # unlabeled_dataset_dir = args.raw_data_path
        # for file in os.listdir(unlabeled_dataset_dir):
        #     data.append({"ori_img": os.path.join(unlabeled_dataset_dir, file)})
        # return data, None
        return np.load(args.raw_data_path, allow_pickle=True), None

    else:
        logging.info("The args.data_type must be 'sup' or 'unsup'.")
        return None, None


def process_and_save_sup_data(args, data):
    file_path = os.path.join(args.output_path,
                             "Labeled_Trainset_path_with_" + str(len(data)) + "_samples.npy")
    np.save(file_path, data)


def process_and_save_unsup_data(args, data):
    unlabeled_data = []
    aug_imge_idx = 0
    for aug_copy_num in range(args.aug_copy):
        ## 增强方法选取
        policy = ImageNetPolicy()
        for item in data:
            # if item.split(".")[-1] == "png":
            #     img = Image.open(item["ori_img"])
            # elif item.split(".")[-1] == "jpg":
            #     img = Image.open(item["ori_img"])
            # else:
            #     logging.info("Your img file format is much seldom..")
            #    img = None
            img = Image.open(item["ori_img"])
            aug_path = os.path.join(args.output_img_data_path,
                                    str(aug_imge_idx) + "_aug.jpg")
            img_aug = policy(img)
            img_aug.save(aug_path)
            aug_imge_idx += 1
            item["aug_img"] = aug_path
            unlabeled_data.append(item)
    save_path = os.path.join(args.output_path,
                             "Unlabeled_Trainset_path_with_" + str(len(unlabeled_data)) + "_.npy")
    np.save(save_path, unlabeled_data)


def main(args):
    if not os.path.exists(args.output_path):
        os.mkdir(args.output_path)

    data, key_dict = load_dataset(args)
    print(data)
    if args.data_type == "sup":
        logging.info("### Processing supervised data ###")
        with open(os.path.join(args.output_path, "image_label_dict.txt"), 'w') as f:
            f.write(str(key_dict))
        process_and_save_sup_data(args, data)

    elif args.data_type == "unsup":
        logging.info("### Processing unsupervised data ###")
        process_and_save_unsup_data(args, data)

    else:
        logging.info("The args.data_type must be 'sup' or 'unsup'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Data augment.")
    parser.add_argument('--data_type', default="sup", type=str,
                        help="Whether to process supervised data or unsupervised data."
                             "sup or unsup")
    parser.add_argument('--raw_data_path',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/test_patch/",
                        type=str,
                        help="The path to the raw data. Using img folder when sup"
                             "and using path file when unsup ")
    parser.add_argument('--output_path',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/data_enrichment/test",
                        type=str,
                        help="The path to save the processed data.")
    parser.add_argument('--output_img_data_path',
                        default="/home/qianslab/yangwenhui/TUMOR_data_processed/bag_files/data_enrichment/test/out",
                        type=str,
                        help="The path to save processed image data.")

    # configs for precessing supervised data
    # parser.add_argument('--sup_size', default=None, type=int,
    # help="Number of supervised pairs to use.")

    # cofings for processing unsupervised data
    parser.add_argument('--aug_copy', default=10, type=int,
                        help="Number of augmented copies to create.")

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)

    main(args)
