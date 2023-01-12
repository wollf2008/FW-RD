from PIL import Image
import os, shutil
import cv2
import numpy as np

good = 0
T_S = 0
N_T_S = 0



def image_resize(img, new_path, new_name):
    h= img.shape[0]
    w = img.shape[1]

    if h>=200 and w>=200:
        resize_img = img[0:200, 0:200]
        cv2.imwrite(os.path.join(new_path, new_name), resize_img)


def MVTEC_name(N):
    if N < 10:  # 确定新文件名
        new_name = '000' + str(N)
    elif N < 100:
        new_name = '00' + str(N)
    elif N < 1000:
        new_name = '0' + str(N)
    else:
        new_name = str(N)
    return new_name


def MVTEC_type_train_imgs_converting(old_path, new_path):
    Tumor = 0
    Stroma = 0
    Normal = 0
    TS = 0
    train_imgs = os.listdir(old_path)

    for train_img in train_imgs:
        if '[1, 0, 0]' in train_img:
            Tumor += 1

        elif '[0, 1, 0]' in train_img:
            Stroma += 1

        elif '[0, 0, 1]' in train_img:

            new_name = MVTEC_name(Normal)
            #
            bgr_img = cv2.imread(os.path.join(old_path, train_img))  # 读取图像
            #
            image_resize(bgr_img, os.path.join(new_path, 'good'), new_name+'.png')

            Normal += 1

        elif '[1, 1, 0]' in train_img:

            new_name = MVTEC_name(TS)
            #
            bgr_img = cv2.imread(os.path.join(old_path, train_img))  # 读取图像
            #
            image_resize(bgr_img, os.path.join(new_path, 'bad'), new_name+'.png')

            TS += 1

def MVTEC_type_test_imgs_converting(old_path_img, old_path_mask, old_path_background_mask, new_path_img, new_path_mask, new_path_background):  # 只针对有Normal的图像进行处理
    test_masks = os.listdir(old_path_mask)
    global good
    global T_S
    global N_T_S

    for test_mask in test_masks:

        N_flg = False
        TS_flg = False
        bgr_img = cv2.imread(os.path.join(old_path_img, test_mask))
        bgr_mask = cv2.imread(os.path.join(old_path_mask, test_mask))  # bgr yellow = [0 152 243]
        bgr_background = cv2.imread(os.path.join(old_path_background_mask, test_mask))
        (h, w, c) = bgr_mask.shape
        new_mask = np.zeros((h, w, 3), np.uint8)

        for i in range(0, h):  # 遍历每个像素
            for j in range(0, w):
                [b, g, r] = bgr_mask[i, j]
                if [b, g, r] == [0, 152, 243]:
                    N_flg = True
                elif [b, g, r] == [128, 64, 0] or [b, g, r] == [0, 128, 64]:
                    TS_flg = True


                if [b, g, r] == [0, 152, 243] or [b, g, r] == [255, 255, 255]:
                    new_mask[i, j] = [0, 0, 0]
                else:
                    new_mask[i, j] = [255, 255, 255]

        if N_flg == True and TS_flg == True:
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)
            if h > 400 or w >400:
                new_img = bgr_img #temp
                new_background = bgr_background
                new_h = new_mask.shape[0]
                new_w = new_mask.shape[1]

                for i in range(int(new_h/200)):
                    for j in range(int(new_w/200)):
                        new_mask_crop = new_mask[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        new_img_crop = new_img[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        new_background_crop = new_background[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        if set(new_mask_crop.flatten()) == {0, 255}:
                            new_name = MVTEC_name(N_T_S)
                            image_resize(new_mask_crop, os.path.join(new_path_mask, 'N+T+S'), new_name + '_mask.png')
                            image_resize(new_img_crop, os.path.join(new_path_img, 'N+T+S'), new_name + '.png')
                            image_resize(new_background_crop, os.path.join(new_path_background, 'N+T+S'), new_name + '_background.png')
                            N_T_S += 1
                        elif set(new_mask_crop.flatten()) == {0}:
                            new_name = MVTEC_name(good)
                            image_resize(new_img_crop, os.path.join(new_path_img, 'good'), new_name + '.png')
                            image_resize(new_background_crop, os.path.join(new_path_background, 'good'), new_name + '_background.png')
                            good += 1

            else:
                new_name = MVTEC_name(N_T_S)
                image_resize(new_mask, os.path.join(new_path_mask, 'N+T+S'), new_name + '_mask.png')
                image_resize(bgr_img, os.path.join(new_path_img, 'N+T+S'), new_name + '.png')
                image_resize(bgr_background, os.path.join(new_path_background, 'N+T+S'), new_name + '_background.png')
                N_T_S += 1


        elif N_flg == True and TS_flg == False:
            if h > 400 or w > 400:
                new_img = bgr_img
                new_background = bgr_background
                new_h = new_img.shape[0]
                new_w = new_img.shape[1]

                for i in range(int(new_h / 200)):
                    for j in range(int(new_w / 200)):
                        new_name = MVTEC_name(good)
                        new_img_crop = new_img[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        new_background_crop = new_background[i * 200:(i + 1) * 200, j * 200: (j + 1) * 200]
                        image_resize(new_img_crop, os.path.join(new_path_img, 'good'), new_name + '.png')
                        image_resize(new_background_crop, os.path.join(new_path_background, 'good'), new_name + '_background.png')
                        good += 1
            else:
                new_name = MVTEC_name(good)
                image_resize(bgr_img, os.path.join(new_path_img, 'good'), new_name + '.png')
                image_resize(bgr_background, os.path.join(new_path_background, 'good'), new_name + '_background.png')
                good += 1

        elif N_flg == False and TS_flg == True:

            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_BGR2RGB)
            new_mask = cv2.cvtColor(new_mask, cv2.COLOR_RGB2GRAY)
            if h > 400 or w > 400:
                new_img = bgr_img
                new_background = bgr_background
                new_h = new_mask.shape[0]
                new_w = new_mask.shape[1]

                for i in range(int(new_h / 200)):
                    for j in range(int(new_w / 200)):
                        new_mask_crop = new_mask[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        new_img_crop = new_img[i * 200:(i+1)*200, j*200: (j + 1) * 200]
                        new_background_crop = new_background[i * 200:(i + 1) * 200, j * 200: (j + 1) * 200]
                        if set(new_mask_crop.flatten()) == {0, 255}:
                            new_name = MVTEC_name(T_S)
                            image_resize(new_mask_crop, os.path.join(new_path_mask, 'T+S'), new_name + '_mask.png')
                            image_resize(new_img_crop, os.path.join(new_path_img, 'T+S'), new_name + '.png')
                            image_resize(new_background_crop, os.path.join(new_path_background, 'T+S'), new_name + '_background.png')
                            T_S += 1

            else:
                new_name = MVTEC_name(T_S)
                image_resize(new_mask, os.path.join(new_path_mask, 'T+S'), new_name + '_mask.png')
                image_resize(bgr_img, os.path.join(new_path_img, 'T+S'), new_name + '.png')
                image_resize(bgr_background, os.path.join(new_path_background, 'T+S'), new_name + '_background.png')
                T_S += 1


if __name__ == '__main__':

    original_image_path = 'original_dataset/WSSS4LUAD'
    patches_path = 'data/WSSS4LUAD'

    if os.path.exists(patches_path):
        print('wsss4luad_mvtec_format already exists')
    else:
        os.makedirs(os.path.join(patches_path,'train/good/'))
        os.makedirs(os.path.join(patches_path, 'train/bad/'))
        os.makedirs(os.path.join(patches_path, 'test/T+S/'))
        os.makedirs(os.path.join(patches_path, 'test/N+T+S/'))
        os.makedirs(os.path.join(patches_path, 'test/good/'))
        os.makedirs(os.path.join(patches_path, 'ground_truth/T+S/'))
        os.makedirs(os.path.join(patches_path, 'ground_truth/N+T+S/'))
        os.makedirs(os.path.join(patches_path, 'background_mask/good/'))
        os.makedirs(os.path.join(patches_path, 'background_mask/T+S/'))
        os.makedirs(os.path.join(patches_path, 'background_mask/N+T+S/'))

    train_path = os.path.join(original_image_path, '1.training')
    test_path_img = os.path.join(original_image_path, '3.testing', 'img')
    test_path_mask = os.path.join(original_image_path, '3.testing', 'mask')
    test_path_background_mask = os.path.join(original_image_path, '3.testing', 'background-mask')
    validation_path_img = os.path.join(original_image_path, '2.validation', 'img')
    validation_path_mask = os.path.join(original_image_path, '2.validation', 'mask')
    validation_path_background_mask = os.path.join(original_image_path, '2.validation', 'background-mask')

    MVTEC_type_train_imgs_converting(train_path,'data/WSSS4LUAD/train')
    MVTEC_type_test_imgs_converting(test_path_img, test_path_mask, test_path_background_mask,os.path.join(patches_path, 'test/'),
                                    os.path.join(patches_path, 'ground_truth/'), os.path.join(patches_path, 'background_mask/'))
    MVTEC_type_test_imgs_converting(validation_path_img, validation_path_mask, validation_path_background_mask,os.path.join(patches_path, 'test/'),
                                    os.path.join(patches_path, 'ground_truth/'), os.path.join(patches_path, 'background_mask/'))
