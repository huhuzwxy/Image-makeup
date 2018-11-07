import dlib
import cv2
import sys
import numpy as np
import face_recognition
from PIL import Image,ImageDraw,ImageEnhance
import math

img_new = Image.open('4.jpg')
img = face_recognition.load_image_file('8.jpg')
img_cv = cv2.imread('6.jpg')
cv2.imshow('image',img_cv)
cv2.waitKey(0)

print(img)
face_landmarks_list = face_recognition.face_landmarks(img)
print(face_landmarks_list)
cv2.imshow('orginal',img_cv)
cv2.waitKey(0)

# 高斯滤波磨皮
img_gauss = np.hstack([cv2.GaussianBlur(img_cv,(7,7),0)])
cv2.imwrite('gauss.jpg',img_gauss)
cv2.imshow('image',img_gauss)
cv2.waitKey(0)

# 全图双边滤波
img_bil = np.hstack([cv2.bilateralFilter(img_cv,9,51,51)])
cv2.imwrite('bil.jpg',img_bil)
cv2.imshow('image_bil',img_bil)
cv2.waitKey(0)
ycrcb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2YCR_CB)
channels = cv2.split(ycrcb)
cv2.equalizeHist(channels[0], channels[0])
cv2.merge(channels, ycrcb)
cv2.cvtColor(ycrcb, cv2.COLOR_YCR_CB2BGR, img_cv)
cv2.imshow('a',img_cv)
cv2.waitKey(0)

# 肤色检测+双边滤波

# 缩放和裁剪
img_resize = img_new.resize((128,128))
img_crop = img_new.crop((80,100,260,300))
img_resize.show()
img_crop.show()

# 旋转镜像
img_rotate0 = img_new.rotate(45)
img_rotate = img_new.transpose(Image.FLIP_LEFT_RIGHT)
img_rotate1 = img_new.transpose(Image.FLIP_TOP_BOTTOM)
img_rotate.show()
img_rotate1.show()

# 饱和度
img_color = ImageEnhance.Color(img_new).enhance(2.0)
img_color.show()

# 亮度
img_bright = ImageEnhance.Brightness(img_new).enhance(0.5)
img_bright1 = ImageEnhance.Brightness(img_new).enhance(1.5)
img_bright.show()
img_bright1.show()

# 对比度
img_contrast = ImageEnhance.Contrast(img_new).enhance(2.0)
img_contrast.show()

# 素描
width,height = img_new.size
img_draw = img_new.convert('L')
pix = img_draw.load()

for w in range(width):
    for h in range(height):
        if w == width -1 or h == height - 1:
            continue
        src = pix[w,h]
        dst = pix[w+1,h+1]
        diff = abs(src - dst)
        if diff >= 10:
            pix[w,h] = 0
        else:
            pix[w,h] = 255

img_draw.show()

for face_landmark in face_landmarks_list:
    facial_features = ['chin','left_eyebrow','right_eyebrow','nose_bridge','nose_tip','left_eye','right_eye','top_lip','bottom_lip']
    pil_img = Image.fromarray(img)
    pil_img.show()
    print(pil_img)
    d = ImageDraw.Draw(pil_img)

    #img1 = color(face_landmark['top_lip'])
    #cv2.imshow('image',img1)
    #for facial_feature in facial_features:
    #    d.line(face_landmark[facial_feature],width = 5)

    #pil_img.show()

    d1 = ImageDraw.Draw(pil_img,'RGBA')

    d1.polygon(face_landmark['top_lip'],fill=(150,30,0,128))
    d1.polygon(face_landmark['bottom_lip'],fill=(150,30,0,128))
    d1.line(face_landmark['top_lip'],fill=(150,60,0,64),width = 0)
    d1.line(face_landmark['bottom_lip'],fill=(150,60,0,64),width = 0)

    pil_img.show()
    pil_img = np.asarray(pil_img)
    pil_img = pil_img[...,::-1]
    cv2.imshow('a',pil_img)
    cv2.imwrite('contrast.jpg',pil_img)
    cv2.waitKey(0)








