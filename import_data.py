import os
import h5py
import numpy as np
import cv2
import random
from PIL import Image

N = 20
channels = 3
height = 584
width = 565


def write_hdf5(arr, outfile):
    with h5py.File(outfile, "w") as f:
        f.create_dataset("image", data=arr, dtype=arr.dtype)


def load_hdf5(infile):
    with h5py.File(infile, "r") as f:
        return f["image"][()]


def tif_to_hdf5(data_type):
    images = np.empty((N, height, width, channels))
    truths = np.empty((N, height, width))
    masks = np.empty((N, height, width))
    for root, dirs, files in os.walk(f'DRIVE/{data_type}/images/'):
        for i in range(N):
            image = cv2.imread(f'DRIVE/{data_type}/images/{files[i]}')
            images[i] = np.array(image)
            truth = Image.open(f'DRIVE/{data_type}/1st_manual/{files[i][0:2]}_manual1.gif')
            truths[i] = np.array(truth)
            mask = Image.open(f'DRIVE/{data_type}/mask/{files[i][0:2]}_{data_type}_mask.gif')
            masks[i] = np.array(mask)
    images = np.transpose(images, (0, 3, 1, 2))
    truths = truths.reshape((N, 1, height, width))
    masks = masks.reshape((N, 1, height, width))
    write_hdf5(images, f'images_{data_type}.hdf5')
    write_hdf5(truths, f'truths_{data_type}.hdf5')
    write_hdf5(masks, f'masks_{data_type}.hdf5')


def data_preprocess(images):
    images = rgb2gray(images)
    images = normalization(images)
    images = clahe_equalized(images)
    images = adjust_gamma(images, 1.2)
    images = images/255
    return images


def rgb2gray(rgb):
    gray = rgb[:, 0, ...] * 0.299 + rgb[:, 1, ...] * 0.587 + rgb[:, 2, ...] * 0.114
    gray = np.reshape(gray, (gray.shape[0], 1, gray.shape[1], gray.shape[2]))
    return gray


def normalization(images):
    std = np.std(images)
    mean = np.mean(images)
    images_normalized = (images - mean) / std
    for i in range(images.shape[0]):
        images_max = np.max(images_normalized[i])
        images_min = np.min(images_normalized[i])
        images_normalized[i] = ((images_normalized[i] - images_min) / (images_max-np.min(images_normalized[i])))
    images_normalized = images_normalized * 255.
    return images_normalized


def clahe_equalized(images):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    for i in range(images.shape[0]):
        images[i, 0] = clahe.apply(images[i, 0].astype(np.uint8))
    return images


def adjust_gamma(images, gamma=1.0):
    inv_gamma = 1.0 / gamma
    table = []
    for i in range(256):
        table.append(((i / 255.0) ** inv_gamma) * 255)
    table = np.array(table).astype(np.uint8)
    for i in range(images.shape[0]):
        images[i, 0] = cv2.LUT(images[i, 0].astype(np.uint8), table)
    return images


def extract_random(images, masks, patch_h, patch_w, patches):
    image_number = images.shape[0]
    channel_number = images.shape[1]
    image_h = images.shape[2]
    image_w = images.shape[3]
    patch_image_number = int(patches / image_number)
    patches_images = np.empty((patches, channel_number, patch_h, patch_w))
    patches_masks = np.empty((patches, channel_number,  patch_h, patch_w))
    count = 0
    for index in range(image_number):
        number = 0
        while number < patch_image_number:
            x_center = random.randint(0+int(patch_w/2), image_w-int(patch_w/2))
            y_center = random.randint(0+int(patch_h/2), image_h-int(patch_h/2))
            patch_img = images[index, :, (y_center-patch_h//2):(y_center+patch_h//2), (x_center-patch_w//2):(x_center+patch_w//2)]
            patch_mask = masks[index, :, (y_center-patch_h//2):(y_center+patch_h//2), (x_center-patch_w//2):(x_center+patch_w//2)]
            patches_images[count] = patch_img
            patches_masks[count] = patch_mask
            count += 1
            number += 1
    return patches_images, patches_masks


def load_train_data_hdf5(patch_h, patch_w, patches):
    train_images = data_preprocess(load_hdf5('images_training.hdf5')) / 255.
    train_masks = load_hdf5('truths_training.hdf5') / 255.
    train_images = train_images[:, :, 9: 574, :]
    train_masks = train_masks[:, :, 9: 574, :]
    patches_images_train, patches_masks_train = extract_random(train_images, train_masks, patch_h, patch_w, patches)
    patches_images_train = patches_images_train.reshape((patches, patch_h, patch_w, 1))
    patches_masks_train = patches_masks_train.reshape((patches, patch_h, patch_w, 1))
    temp = np.zeros((patches, patch_h, patch_w, 2))
    for i in range(patches):
        for j in range(patch_h):
            for k in range(patch_w):
                if patches_masks_train[i, j, k, 0] == 1:
                    temp[i, j, k, 0] = 0
                    temp[i, j, k, 1] = 1
                else:
                    temp[i, j, k, 0] = 1
                    temp[i, j, k, 1] = 0
    patches_masks_train = temp
    return patches_images_train, patches_masks_train


def pad_overlap(imgs, patch_h, patch_w, stride_h, stride_w):
    N_imgs = imgs.shape[0]
    N_channels = imgs.shape[1]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    pad_h = stride_h - (img_h-patch_h)%stride_h
    pad_w = stride_w - (img_w-patch_w)%stride_w
    if ((img_h-patch_h)%stride_h != 0):
        new_imgs = np.zeros((N_imgs, N_channels, img_h+pad_h, img_w))
        new_imgs[:,:,0:img_h,0:img_w] = imgs
        imgs = new_imgs
    if ((img_w-patch_w)%stride_w != 0):
        new_imgs = np.zeros((N_imgs, N_channels, imgs.shape[2], img_w+pad_w))
        new_imgs[:,:,:,0:img_w] = imgs
        imgs = new_imgs
    return imgs


def extract_ordered_overlap(imgs, patch_h, patch_w, stride_h, stride_w):
    N_imgs = imgs.shape[0]
    N_channels = imgs.shape[1]
    img_h = imgs.shape[2]
    img_w = imgs.shape[3]
    N_patch_h = (img_h-patch_h)//stride_h+1
    N_patch_w = (img_w-patch_w)//stride_w+1
    N_patch_img = N_patch_h * N_patch_w
    N_patch_total = N_patch_img * N_imgs
    patches = np.empty((N_patch_total, N_channels, patch_h, patch_w))
    count = 0
    for i in range(N_imgs):
        for h in range(N_patch_h):
            for w in range(N_patch_w):
                patch = imgs[i, :, (h*stride_h):((h*stride_h)+patch_h), (w*stride_w):((w*stride_w)+patch_w)]
                patches[count] = patch
                count +=1
    return patches


def load_test_data_hdf5(patch_h, patch_w, stride_h, stride_w):
    test_images = data_preprocess(load_hdf5('images_test.hdf5'))
    test_truths = load_hdf5('truths_test.hdf5') / 255.
    test_images = pad_overlap(test_images, patch_h, patch_w, stride_h, stride_w)
    patches_images_test = extract_ordered_overlap(test_images, patch_h, patch_w, stride_h, stride_w)
    patches_images_test = patches_images_test.reshape((patches_images_test.shape[0], patch_h, patch_w, patches_images_test.shape[1]))
    return patches_images_test, test_images.shape[2], test_images.shape[3], test_truths


if __name__ == '__main__':
    tif_to_hdf5('training')
    tif_to_hdf5('test')
