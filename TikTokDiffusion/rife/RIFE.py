import glob
import os
import time
import re
import cv2
import torch
from torch.nn import functional as F
import warnings
from train_log.RIFE_HDv3 import Model
from itertools import pairwise

model = None
modeldir = r"TikTokDiffusion\rife\train_log"
other_modeldir = r"rife\train_log"


def load_model():
    global model
    model = Model()
    try:
        model.load_model(modeldir, -1)
    except FileNotFoundError:
        model.load_model(other_modeldir, -1)


def grouper(iterable, n):
    args = [iter(iterable)] * n
    return zip(*args)


def generate_frames(start_image, end_image, generate_image_num, save_folder):

    save_images = []
    warnings.filterwarnings("ignore")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    ratio = 0
    rthreshold = 0.02
    rmaxcycles = 8
    ratio = 0
    generate_image_num = generate_image_num+1

    model = Model()
    try:
        model.load_model(modeldir, -1)
    except FileNotFoundError:
        model.load_model(other_modeldir, -1)

    if not hasattr(model, 'version'):
        model.version = 0

    model.eval()
    model.device()

    img0 = cv2.imread(start_image, cv2.IMREAD_UNCHANGED)
    img1 = cv2.imread(end_image, cv2.IMREAD_UNCHANGED)
    height, width, channels = img0.shape

    img0 = cv2.resize(img0, (width, height))
    img1 = cv2.resize(img1, (width, height))

    img0 = (torch.tensor(img0.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)
    img1 = (torch.tensor(img1.transpose(2, 0, 1)).to(device) / 255.).unsqueeze(0)

    n, c, h, w = img0.shape
    ph = ((h - 1) // 64 + 1) * 64
    pw = ((w - 1) // 64 + 1) * 64
    padding = (0, pw - w, 0, ph - h)
    img0 = F.pad(img0, padding)
    img1 = F.pad(img1, padding)

    if ratio:
        if model.version >= 3.9:
            img_list = [img0, model.inference(img0, img1, ratio), img1]
        else:
            img0_ratio = 0.0
            img1_ratio = 1.0
            if ratio <= img0_ratio + rthreshold / 2:
                middle = img0
            elif ratio >= img1_ratio - rthreshold / 2:
                middle = img1
            else:
                tmp_img0 = img0
                tmp_img1 = img1
                for inference_cycle in range(rmaxcycles):
                    middle = model.inference(tmp_img0, tmp_img1)
                    middle_ratio = (img0_ratio + img1_ratio ) / 2
                    if ratio - (rthreshold / 2) <= middle_ratio <= ratio + (rthreshold / 2):
                        break
                    if ratio > middle_ratio:
                        tmp_img0 = middle
                        img0_ratio = middle_ratio
                    else:
                        tmp_img1 = middle
                        img1_ratio = middle_ratio
            img_list.append(middle)
            img_list.append(img1)
    else:
        if model.version >= 3.9:
            img_list = []
            for i in range(generate_image_num-1):
                img_list.append(model.inference(img0, img1, (i+1) * 1. / generate_image_num))
        else:
            img_list = [img0, img1]
            for i in range(exp):
                tmp = []
                for j in range(len(img_list) - 1):
                    mid = model.inference(img_list[j], img_list[j + 1])
                    tmp.append(img_list[j])
                    tmp.append(mid)
                tmp.append(img1)
                img_list = tmp

    for i in range(len(img_list)):
        save_images.append((img_list[i][0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
    return save_images


def get_missing_imgs(path_to_search) -> tuple[list[list[str]], int]:
    all_imgs = glob.glob(path_to_search + "\*.png")
    all_imgs_paired = pairwise(all_imgs)
    missing_pairs = []
    zfill_amount = 0
    for first_img, second_img in all_imgs_paired:

        first_img_stripped = first_img.replace(".png", "")
        first_found_pattern = re.findall("\d+$", first_img_stripped)[0]
        first_number_in_name = int(first_found_pattern)

        second_img_stripped = second_img.replace(".png", "")
        second_found_pattern = re.findall("\d+$", second_img_stripped)[0]
        second_number_in_name = int(second_found_pattern)

        delta = second_number_in_name-first_number_in_name-1
        if delta > 0:
            missing_pairs.append([first_img, second_img, delta])

    zfill_amount = len(str(first_found_pattern))
    return missing_pairs, zfill_amount


def frame_generator(path_of_images, save_folder):
    missing_pairs, zfill_amount = get_missing_imgs(path_of_images)
    images_to_save = []
    load_model()
    print("generating inbetween frames")

    for image1, image2, missing_num in missing_pairs:
        file_names = []
        img1_stripped = image1.replace(".png", "")
        first_number_in_name = int(re.findall("\d+$", img1_stripped)[0])

        img2_stripped = image2.replace(".png", "")
        second_number_in_name = int(re.findall("\d+$", img2_stripped)[0])

        delta = second_number_in_name - first_number_in_name - 1

        for count in range(delta):
            count += 1
            name = first_number_in_name + count
            name = str(name).zfill(zfill_amount)+".png"

            file_names.append(name)

        generated_frames = generate_frames(image1, image2, generate_image_num=missing_num, save_folder=save_folder)
        images_to_save.append(zip(generated_frames, file_names))

    for zip_object in images_to_save:
        for file, file_name in zip_object:
            if not os.path.exists(save_folder):
                os.mkdir(save_folder)
            file_name = os.path.join(save_folder, file_name)
            cv2.imwrite(file_name, file)


