# -*-coding:utf-8-*-
'''
文档主要集成了处理噪音的方法

'''
from PIL import Image, ImageDraw, ImageFont
from PIL import Image
import cv2
import numpy as np
import imutils
import string
import random
import glob, os
from os import listdir
from os.path import isfile, join
import skimage.util
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
# import scipy.stats


#GET NOISE LEVEL
'''
根据高斯分布随机生成一个噪声级别，一个噪声级别对应一种产生噪音的效果

'''
def Get_Noise_Level(level):

    levels = len(level)



#File2List
'''
读取文件并生成list
params
path：文件路径


returns：
返回文字list
'''
def File2List(path):
    f = open(path, 'r')
    line = f.readline()
    str = ''
    while line:
        line = f.readline()
        str += line
    f.close()
    str = str.split('\n')
    str = ''.join(str)
    list1 = list(set(str)) #去除重复的字符
    list1.remove('（')     #去除括号
    list1.remove('）')
    list1.remove(' ')
    return list1



#GetSamplePic
'''
将样本字体生成图片并且存储
params
list 导入的文字list
font 导入的字体
path 输出的路径
H 图片底的高度
W 图片底的宽度
'''
def GetSamplePic(list,font,path,H,W):
    for char in list:
        char_pos = (0, 0)
        img = Image.new('RGB', (H, W), (255, 255, 255))
        draw = ImageDraw.Draw(img)
        draw.text((10,10),char,(0),font=font)
        img.save(path + '/'+char + '.png')
    return 0


#Get_Edges
'''
将原始字体图片进行左右、上下的切割。达到上下左右顶格效果
params
img 导入图片文件

returns
返回已经处理好的图片
'''
def get_edges(img1):
    H,W = img1.shape
    # img1 = cv2.Canny(img,50,150)
    img_H, img_W = img1.shape
    cut_r,cut_l,cut_t,cut_b = [0,0,0,0]
    #cut right
    for i in range(0,img_W,2):
        pixel_sum = np.sum(img1[0:img_H, img_W-i-2:img_W-i])
        if pixel_sum >250:
            cut_r = i
            break
    #cut left
    for i in range(0, img_W, 2):
        pixel_sum = np.sum(img1[0:img_H, i:i+2])
        if pixel_sum > 250:
            cut_l = i
            break
    #cut top
    for i in range(0, img_H, 2):
        pixel_sum = np.sum(img1[i:i + 2,0:img_W])
        if pixel_sum > 250:
            cut_t = i
            break
    #cut bottom
    for i in range(0, img_H, 2):
        pixel_sum = np.sum(img1[ img_H-i-2:img_H-i,0:img_W])
        if pixel_sum > 250:
            cut_b = i
            break
    img_result = img1[cut_t:H-cut_b,cut_l:img_W - cut_r]


    return img_result

#Compress and Fetch for x
def xCompress_Fetch(img,level):
    H,W = img.shape
    img = cv2.resize(img,(H,round(W * level)),interpolation=cv2.INTER_AREA)
    return img

#Compress and Fetch for y
def yCompress_Fetch(img,level):
    H,W = img.shape
    img = cv2.resize(img,(round(H*level),W ),interpolation=cv2.INTER_AREA)
    return img



#dilate
def Dilate(img,level):

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (round(level), round(level)))
    img = cv2.dilate(img, kernel)


    return img


#erode
def Erode(img,level):
    kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (round(level), round(level)))
    img = cv2.erode(img, kernel1)
    return img


#adding noise
def noise(img,level):
    return img

#process_img
def process_img(img,compress1,compress2,dilate,erode):
    # print (compress1)
    img = xCompress_Fetch(img,compress1)
    # plot_image(img)
    # print (compress1)
    img = yCompress_Fetch(img,compress2)
    # plot_image(img)
    img = Dilate(img, dilate)
    # plot_image(img)
    img = Erode(img, erode)
    # plot_image(img)
    return img
#Resize
'''
固定并生成要求像素大小的图片（最后在上下左右各加上4pixel的黑边）
params
img:输入的图片文件
H:resize 图片的高度
W:resize 图片的宽度

retruns
返回处理图片
'''
def resize(img,H,W):
    img_H, img_W = img.shape
    padded_img = np.zeros((H, W), dtype=np.uint8) #
    # padded_base = np.zeros((H+8,W+8),dtype=np.uint8)
    if img_H >= img_W:                  #如果高长，则以高为主
        img = imutils.resize(img, height=H)
        w_pos = (W - img.shape[1]) // 2
        padded_img[:, w_pos:w_pos + img.shape[1]] = 255 - img
    else:                               #如果宽长，则以宽为主
        img = imutils.resize(img, width=W)
        h_pos = (H - img.shape[0]) // 2
        padded_img[h_pos:h_pos + img.shape[0], :] = 255 - img
    img = padded_img
    # img = cv2.adaptiveThreshold(255 - img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                            cv2.THRESH_BINARY, 11, 5)
    #    img = 255 - img
    H1, W1 = img.shape
    print (H1,W1)
    # padded_base[4:H1+4,4:W1+4] = img
    # img_final = padded_base
    return img

#RotatePic
'''
将图片按照要求角度进行旋转
params
img: 输入图片
angle:旋转角

returns
返回处理图片

'''
def RotatePic(img, angle):
    H,W = img.shape
    rotated = imutils.rotate(img, angle)
    # rot_mat = cv2.getRotationMatrix2D((H/2, W/2), angle, 1.0)
    # img2 = resize(img, 56, 56)
    # img2 = cv2.warpAffine(img, rot_mat, (H, W), flags=cv2.INTER_LINEAR)
    return rotated


#Processing
'''
处理输入图片，包括切割，resize，旋转，之后生成训练基础样本（未加噪声版本）。
params
path:存储样本的文件夹路径
dest:输出文件夹路径
angles:旋转的角度范围 （-angles,+angles）


'''
def Processing(path,dest,angles):
    #find directory and list all pictures
    dir = path
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    #deal with each pic
    for file in onlyfiles:
        if '.png' in file:
            directory_name = file[:len(file)-4]
            dir_tmp = dir + '/' + file
            directory_path = dest+ '/' +directory_name   #output address
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            img = cv2.imread(dir_tmp,0)
            img = get_edges(img)
            img = resize(img,56,56)
            for angle in range(-angles,angles,1):
                filename = directory_name + str(angle)
                filepath = directory_path + '/' + filename +'.png'
                imgtmp = RotatePic(img,angle)
                imgtmp = 255 - imgtmp                           #do the adaptive threshold
                imgtmp = 255 - cv2.adaptiveThreshold(imgtmp, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                  cv2.THRESH_BINARY, 11, 2)
                cv2.imwrite(filepath,imgtmp)



#RegionNoise
'''
在一定的区域內，按照高斯分布加入噪声
params：

image ： 输入图片
prob ： 阈值，加入噪声的判断标准

returns

输出那部分的区域图片

'''

def gs_noise(image,prob):

    output = np.zeros(image.shape,np.uint8)
    thres = 1 - prob
    for i in range(0,image.shape[0],2):
        for j in range(0,image.shape[1],2):
            rdn = random.random()
            if rdn < prob:
                output[i:i+2,j:j+2] = 0

            elif rdn > thres:
                output[i:i + 2, j:j + 2] = 0
            else:
                output[i:i + 2, j:j + 2] = image[i:i+2,j:j+2]
    return output



#adding_noise
'''
对图片的区域动态的加入噪声。固定一个filter，并使其在图片中滑动采集pixel的总和，如果总和较大则加入更多的噪声
params
image:输入图片
prob1，2，3:不同像素总和的区域所对应的不同阈值
noise1，2，3:不同的 区域像素／图片像素总和 的比例界限

return
输出图片

'''
def adding_noise(image,prob1,prob2,prob3,noise1,noise2,noise3):
    output = np.zeros(image.shape, np.uint8)
    pixel_sum = np.sum(image)
    print (pixel_sum)
    R_X,R_Y = [16,16]
    # thres = 1-prob
    output = np.zeros(image.shape, np.uint8)
    for i in range(0,image.shape[0],R_X):

        for j in range(0,image.shape[1],R_Y):

            rdn = random.random()
            region = np.zeros((16,16), np.uint8)
            region = image[i:i+R_X,j:j+R_Y]
            region_ratio = np.sum(region)/pixel_sum
            print(region_ratio)
            if(region_ratio > noise1):
                region = gs_noise(region,prob1)
                print('adding')
                output[i:i+R_X,j:j+R_Y] = region
            elif(region_ratio > noise2):
                region = gs_noise(region,prob2)
                print('adding')
                output[i:i+R_X,j:j+R_Y] = region
            elif(region_ratio > noise3):
                region = gs_noise(region, prob3)
                print('adding')
                output[i:i + R_X, j:j + R_Y] = region
            else:
                output[i:i + R_X, j:j + R_Y] = region


    return output

#ADDNoise
'''
对同一图片进行不同标准的噪声处理，并且生成图片集
params
dir: 存储图片集的文件夹
prob：prob的梯度集合
noise：noise的梯度集合
kernel_size：dilate中的kernel的矩阵大小


'''
def AddNoise(dir,prob,noise,kernel_size):
    #get all the directories
    dirlist = [name for name in os.listdir(dir) if os.path.isdir(join(dir, name))]
    prob1t,prob2t,prob3t = prob
    noise1t,noise2t,noise3t = noise
    for direct in dirlist:
        #get the word list
        directory = dir + '/' +direct
        onlyfiles = [f for f in listdir(directory) if isfile(join(directory, f))]
        for file in onlyfiles:
            filename = file[:len(file)-4]
            filepath =  directory + '/' +  file
            img = cv2.imread(filepath,0)
            for i in range(0,3):
                noise_sub = noise[i]
                noise_sub1 = noise_sub
                noise_sub2 = noise_sub1/3
                noise_sub3 = noise_sub2/3
                for j in range(0,3):
                    prob_sub = prob[j]
                    prob_sub1 = prob_sub
                    prob_sub2 = prob_sub1/3
                    prob_sub3 = prob_sub2/3
                    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(kernel_size,kernel_size))
                    dilates1 = cv2.dilate(img, kernel)
                    output = adding_noise(dilates1, prob_sub1,prob_sub2, prob_sub3,noise_sub1, noise_sub2, noise_sub3)
                    outputpath = directory + '/' +filename + 'type' + str(i) +str(j)+'.png'
                    cv2.imwrite(outputpath,output)



'''
显示图像
'''
def plot_image(img):
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()




'''
获取数字骨架
'''
def get_skeleton(img):
    size = np.size(img)
    skel = np.zeros(img.shape, np.uint8)
    #
    # ret,img = cv2.threshold(img,127,255,0)
    # img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                      cv2.THRESH_BINARY, 11, 5)
    # img = cv2.resize(img,(55,55),interpolation=cv2.INTER_AREA)
    ret, img = cv2.threshold(img, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    done = False

    while (not done):
        eroded = cv2.erode(img, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(img, temp)
        skel = cv2.bitwise_or(skel, temp)
        img = eroded.copy()

        zeros = size - cv2.countNonZero(img)
        if zeros == size:
            done = True
    return skel


def clean_char(img):
    H, W = img.shape
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                             cv2.THRESH_BINARY,21, 5) #27 3
    img = 255 - img
    img_sum = img.sum(1)
    start = -1
    end = -1
    for pos, val in enumerate(img_sum):
        if val > 0:
            if start < 0:
                start = pos
        if img_sum[-pos] > 0:
            if end < 0:
                end = H - pos

    return 255 - img[start:end]


"""
获取displacement distortion
"""
def elastic_transform(image, alpha, sigma, random_state=None):
    """Elastic deformation of images as described in [Simard2003]_.
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
       Convolutional Neural Networks applied to Visual Document Analysis", in
       Proc. of the International Conference on Document Analysis and
       Recognition, 2003.
    """
    assert len(image.shape) == 2

    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha

    x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), indexing='ij')
    indices = np.reshape(x + dx, (-1, 1)), np.reshape(y + dy, (-1, 1))

    return map_coordinates(image, indices, order=1,mode='reflect').reshape(shape)


#adding_single noise
'''
Adding noise to get a single picture
elastic distort + blur
params:
img: the input image
sigma: variance
alpha: the cross rate
blur: the blur kernel
path: the address to save the file
'''

def Adding_single_noise(img,s,a,b):

    img1 = img.copy()

    img1 = 255 - elastic_transform(img1, a, s, random_state=None)
    img1 = cv2.GaussianBlur(img1, (b, b), 0)
    # plot_image(img1)
    return img1






#adding noise
'''
Adding noise
elastic distort + blur
params:
img: the input image
sigma: variance
alpha: the cross rate
blur: the blur kernel
path: the address to save the file
'''
def Adding_noise(img,sigma,sigma1,alpha,blur,blur1,path):
    img1 = img.copy()
    for a in alpha:
        print(a)
        if (a != 8):
            for s in sigma:
                # print(s)
                for b in blur:
                    # print(b)
                    img1 = 255 - elastic_transform(img, a, s, random_state=None)
                    img1 = cv2.GaussianBlur(img1, (b, b), 0)
                    plot_image(img1)

        else:
            for s in sigma1:
                # print(s)
                for b in blur1:
                    # print(b)
                    img1 = 255 - elastic_transform(img, a, s, random_state=None)
                    img1 = cv2.GaussianBlur(img1, (b, b), 0)
                    plot_image(img1)
    return img1











# noise = [0.12,0.1,0.08]
# prob = [0.27,0.21,0.12]
# dir = './process'
# AddNoise(dir,prob,noise,3)
compress = [0.8,0.9,1.0,1.1,1.2,1.3]
kernel = [1,3]
#
#
# img = cv2.imread('速.png',0)
#
# img = get_edges(img)
#
# img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
#                                      cv2.THRESH_BINARY, 17, 5)

# for x in compress:
#     for y in compress:
#         for dilate in kernel:
#             for erode in kernel:
#                 imgtmp = img.copy()
#                 imgtmp = process_img(imgtmp,x,y,dilate,erode)
#
#
#                 for angle in range(-2, 2, 1):
#                     imgtmp1 = imgtmp
#                     imgtmp1 = RotatePic(imgtmp1, angle)
#
#                     imgtmp1 = resize(255 - imgtmp1, 56, 56)
#
#                     imgtmp1 = get_skeleton(imgtmp1)
#                     print ('skel')
#                     plot_image(imgtmp1)

#



def pad_image(img, H, W):
    """ white padding an image to expected size
    Args:
        (np.array) img: image
        (int) H: expected height
        (int) W: expected width
    Returns:
        (np.array) all the blocks stacked in a np array
    """
    img = clean_char(img)

    img_H, img_W = img.shape[:2]
    padded_img = np.zeros((H, W), dtype=np.uint8)
    if img_H >= img_W:
        img =imutils.resize(img, height=H)
        w_pos = (W - img.shape[1]) // 2
        padded_img[:, w_pos:w_pos+img.shape[1]] = 255 - img

    else:
        img =imutils.resize(img, width=W)
        h_pos = (H - img.shape[0]) // 2
        padded_img[h_pos:h_pos+img.shape[0], :] = 255 - img

    # plot_image(padded_img)
    return padded_img


def ProcessingForSkeletons(path,dest,angles,compress,kernel,sigma,sigma1,alpha,blur,blur1):
    #find directory and list all pictures
    dir = path
    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f))]
    #deal with each pic
    for file in onlyfiles:
        if '.png' in file:
            directory_name = file[:len(file)-4]
            dir_tmp = dir + '/' + file
            directory_path = dest+ '/' +directory_name   #output address
            if not os.path.exists(directory_path):
                os.makedirs(directory_path)
            img = cv2.imread(dir_tmp,0)
            # img = get_edges(img)
            # img = 255 - cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            #                                   cv2.THRESH_BINARY, 17, 5)
            print  (directory_name)
            for x in compress:
                for y in compress:
                    for dilate in kernel:
                        for erode in kernel:
                            imgtmp = img.copy()
                            imgtmp = process_img(255 - imgtmp, x, y, dilate, erode)

                            for angle in range(-angles, angles, 1):
                                # print (angle)
                                filename = directory_name +str(x) + str(y)  + str(dilate) + str(erode) + str(angle)
                                filepath = directory_path + '/' + filename + '.png'
                                imgtmp1 = 255 - imgtmp
                                # plot_image(255 - imgtmp1)
                                imgtmp1 = RotatePic(255 - imgtmp1, angle)
                                imgtmp1 = 255 - imgtmp1
                                # add noise

                                imgtmp2 = get_edges(255 - imgtmp1)

                                imgtmp2 = pad_image(255 - imgtmp2, 64, 64)
                                cv2.imwrite(filepath, imgtmp2)

                                for a in alpha:
                                        # print(a)
                                        for s in sigma:
                                            # print(s)
                                            for b in blur:
                                                filenamex = directory_name + str(x) + str(y) + str(dilate) + str(
                                                    erode) + str(angle)+str(a)+str(s)+str(b)
                                                filepathx = directory_path + '/' + filenamex + '.png'
                                                imgtmpx = Adding_single_noise(imgtmp1,s,a,b)

                                                imgtmpx = get_edges(imgtmpx)

                                                imgtmpx = pad_image(255 - imgtmpx, 64, 64)

                                                cv2.imwrite(filepathx, imgtmpx)




                                # imgtmp1 = get_edges(imgtmp1)
                                # plot_image(imgtmp1)
                                # imgtmp1 = pad_image(255 - imgtmp1,64,64)
                                # plot_image(imgtmp1)
                                # imgtmp1 = get_skeleton(imgtmp1)
                                # plot_image(imgtmp1)

                                # cv2.imwrite(filepath, imgtmp1)
                                # print ('skel')


#
blur = [1,3,5,7]
blur1 = [5,7,9]
alpha = [2]
sigma = [0.8,1.2]
sigma1 = [1.0,1.1,1.2]
ProcessingForSkeletons('/Users/a00/PycharmProjects/InvoiceReco/sample','./noise4',2,compress,kernel,sigma,sigma1,alpha,blur,blur1)
print ('done!')

#
# shang = cv2.imread('三.png',0)
#
# shang = cv2.resize(shang,(100,120),interpolation=cv2.INTER_AREA)
# plot_image(shang)
# shang = RotatePic(255 - shang,2)
# plot_image(shang)
# shang = get_edges(shang)
# shang = pad_image(255 - shang,64,64)
# shang = get_skeleton(shang)
# plot_image(shang)
img = cv2.imread('速.png',0)
#
# img = pad_image(get_edges(img),56,56)
# #img1 = skimage.util.random_noise(img, mode='localvar', seed=None,clip=True)
# plot_image(img)
# s = 0.5
blur = [1,3,5,7]
blur1 = [5,7,9]
alpha = [2,4]
sigma = [0.8,1.2]
# sigma1 = [1.0,1.1,1.2]

# for a in alpha:
#     print(a)
#     if(a!=8):
#         for s in sigma:
#             # print(s)
#             for b in blur:
#                 # print(b)
#                 img1 = 255 - elastic_transform(img,a,s,random_state=None)
#                 img1 = cv2.GaussianBlur(img1,(b,b),0)
#                 plot_image(img1)
#
#     else:
#         for s in sigma1:
#             # print(s)
#             for b in blur:
#                 # print(b)
#                 img1 = 255 - elastic_transform(img,a,s,random_state=None)
#                 img1 = cv2.GaussianBlur(img1,(b,b),0)
#                 plot_image(img1)



# # Adding_noise(img,sigma,alpha,blur)
#
# for x in compress:
#     for y in compress:
#         for dilate in kernel:
#             for erode in kernel:
#                 imgtmp = img.copy()
#                 imgtmp = process_img(255 - imgtmp, x, y, dilate, erode)
#
#                 for angle in range(-2, 2, 1):
#
#                     imgtmp1 = 255 - imgtmp
#                     # plot_image(255 - imgtmp1)
#                     imgtmp1 = RotatePic(255 - imgtmp1, angle)
#                     # plot_image(imgtmp1)
#                     Adding_noise(imgtmp1,sigma,sigma1,alpha,blur,blur1,'')