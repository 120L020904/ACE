import argparse
import os
from typing import Tuple

import PIL.Image as Image
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
import cv2

def resize_by_width(infile, image_size):
    """按照宽度进行所需比例缩放"""
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = int(x // lv)
    y_s = int(y // lv)
    # print("x_s", x_s, y_s)
    out = im.resize((x_s, y_s), Image.LANCZOS)
    return out


def get_new_img_xy(infile, image_size):
    """返回一个图片的宽、高像素"""
    im = Image.open(infile)
    (x, y) = im.size
    lv = round(x / image_size, 2) + 0.01
    x_s = x // lv
    y_s = y // lv
    # print("x_s", x_s, y_s)
    # out = im.resize((x_s, y_s), Image.ANTIALIAS)
    return x_s, y_s


# 定义图像拼接函数
def image_compose(image_colnum, image_size, image_rownum, image_names, image_save_path, x_new, y_new):
    to_image = Image.new('RGB', (image_colnum * x_new, image_rownum * y_new))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    total_num = 0
    for y in range(1, image_rownum + 1):
        for x in range(1, image_colnum + 1):
            from_image = resize_by_width(image_names[image_colnum * (y - 1) + x - 1], image_size)
            # from_image = Image.open(image_names[image_colnum * (y - 1) + x - 1]).resize((image_size,image_size ), Image.ANTIALIAS)
            to_image.paste(from_image, ((x - 1) * x_new, (y - 1) * y_new))
            total_num += 1
            if total_num == len(image_names):
                break
    return to_image.save(image_save_path)  # 保存新图


def get_image_list_fullpath(dir_path, num_sample):
    # get images
    file_name_list = os.listdir(dir_path)
    file_name_list.sort()
    image_fullpath_list_list = []
    image_fullpath_list = []
    for i, file_name in enumerate(file_name_list):
        if i % num_sample == 0 and i != 0:
            image_fullpath_list_list.append(image_fullpath_list)
            image_fullpath_list = []
        file_one_path = os.path.join(dir_path, file_name)
        if os.path.isfile(file_one_path):
            image_fullpath_list.append(file_one_path)
        else:
            img_path_list = get_image_list_fullpath(file_one_path, num_sample)
            image_fullpath_list.extend(img_path_list)
    image_fullpath_list_list.append(image_fullpath_list)
    return image_fullpath_list_list


def merge_images(image_dir_path, image_save_path, image_size, image_colnum, num_sample):
    # 获取图片集地址下的所有图片名称
    image_fullpath_list_list = get_image_list_fullpath(image_dir_path, num_sample=num_sample)
    # print("image_fullpath_list", len(image_fullpath_list), image_fullpath_list)
    for i, image_fullpath_list in enumerate(image_fullpath_list_list):
        # image_save_path = r'{}.jpg'.format(image_dir_path)  # 图片转换后的地址
        # image_rownum = 4  # 图片间隔，也就是合并成一张图后，一共有几行
        save_path = image_save_path.format(i)
        image_rownum_yu = len(image_fullpath_list) % image_colnum
        if image_rownum_yu == 0:
            image_rownum = len(image_fullpath_list) // image_colnum
        else:
            image_rownum = len(image_fullpath_list) // image_colnum + 1

        x_list = []
        y_list = []
        for img_file in image_fullpath_list:
            img_x, img_y = get_new_img_xy(img_file, image_size)
            x_list.append(img_x)
            y_list.append(img_y)

        # print("x_list", sorted(x_list))
        # print("y_list", sorted(y_list))
        x_new = int(x_list[len(x_list) // 5 * 4])
        y_new = int(y_list[len(y_list) // 5 * 4])
        # print(" x_new, y_new", x_new, y_new)
        image_compose(image_colnum, image_size, image_rownum, image_fullpath_list, save_path, x_new,
                      y_new)  # 调用函数
        # for img_file in image_fullpath_list:
        #     resize_by_width(img_file,image_size)

def text_under_image(image: np.ndarray, text: str, text_color: Tuple[int, int, int] = (0, 0, 0)):
    h, w, c = image.shape
    offset = int(h * .2)
    img = np.ones((h + offset, w, c), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    # font = ImageFont.truetype("/usr/share/fonts/truetype/noto/NotoMono-Regular.ttf", font_size)
    img[:h] = image
    textsize = cv2.getTextSize(text, font, 1, 2)[0]
    text_x, text_y = (w - textsize[0]) // 2, h + offset - textsize[1] // 2
    cv2.putText(img, text, (text_x, text_y ), font, 1, text_color, 2)
    return img
def merge_images_different_model():
    concept_path = ""
    model_name_format = {"SD-v1-4": "",
                       "ESD": "",
                       "SPM": "",
                       "MACE": ""}
    model_output_path = {"SD-v1-4": "",
                  "ESD": "",
                  "SPM": "",
                  "MACE": ""}
    evaluation_dir = "evaluation-outputs/cartoon_eval_format_512"
    image_colnum = 1
    image_rownum = len(model_name_format) // image_colnum
    with open(concept_path, "r") as f:
        for line in f:
            concept = line.strip()
            for key in model_output_path.keys():
                model_output_path[key] = model_name_format[key].format(concept)

    return



if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='generateImages',
        description='Generate Images using Diffusers Code')
    parser.add_argument('--model_name', help='name of model', type=str, required=True)
    parser.add_argument('--csv_name', help='name of csv', type=str, required=True)
    parser.add_argument('--save_path', help='folder where to save images', type=str, required=False)
    parser.add_argument('--is_full', action='store_true', required=False, default=False)
    parser.add_argument('--image_size', help='image size used to train', type=int, required=False, default=256)
    parser.add_argument('--prefix', type=str, required=False, default='')
    parser.add_argument('--add_name', type=str, required=False, default='')
    parser.add_argument('--num_sample', type=int, required=False, default=5)
    parser.add_argument('--concept_path', type=str, required=False, default=None)
    args = parser.parse_args()
    dir_path = os.path.join(args.prefix, "evaluation-outputs", args.csv_name + args.add_name)
    df = pd.read_csv(f"data/concept_csv/{args.csv_name}.csv")
    num_sample = args.num_sample
    concepts = set()
    if args.concept_path is None:
        for _, row in df.iterrows():
            concepts.add(row.concept)
    else:
        with open(args.concept_path, "r") as f:
            for line in f:
                concepts.add(line.strip())
    if args.is_full:
        dir_name_list = os.listdir(dir_path)
        for dir_name in tqdm(dir_name_list):
            for i, concept in enumerate(concepts):
                image_dir_path = os.path.join(dir_path, dir_name, concept)  # 图片集地址
                image_save_path = os.path.join(dir_path, dir_name, concept + "_{}.png")
                image_size = args.image_size  # 每张小图片的大小
                image_colnum = 5  # 合并成一张图后，一行有几个小图
                if not os.path.exists(image_save_path.format(0)):
                    merge_images(image_dir_path, image_save_path, image_size, image_colnum, num_sample)
    else:
        dir_name = args.model_name
        for concept in concepts:
            image_dir_path = os.path.join(dir_path, dir_name, concept)  # 图片集地址
            image_save_path = os.path.join(dir_path, dir_name, concept + "_{}.png")
            image_size = args.image_size  # 每张小图片的大小
            image_colnum = 5  # 合并成一张图后，一行有几个小图
            merge_images(image_dir_path, image_save_path, image_size, image_colnum, num_sample)
