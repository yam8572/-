#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名

# data visualisation and manipulation
import matplotlib.pyplot as plt #show img and table

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity


# In[2]:


def get_mean_and_std(typeStr, img_name, x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    
    # 印出平均與標準差
    if(typeStr == 's'):
        print("\n來源圖檔名: ", img_name)
    elif(typeStr == 't'):
        print("\n目標圖檔名: ", img_name)
    
    print("RGB_channel 平均: ", x_mean)
    print("RGB_channel 標準差: ", x_std)
    
    return x_mean, x_std


# In[3]:


def color_transfer(src_rgb, tar_rgb, src_name, tar_name, coltra_name, coltra_num):
    
    s_mean, s_std = get_mean_and_std('s', src_name, src_rgb)
    t_mean, t_std = get_mean_and_std('t', tar_name, tar_rgb)
    
    # 輸出 source target 的 mean and std txt 檔
    output_mean_and_std_txt(coltra_name, s_mean, s_std, t_mean, t_std, coltra_num)

    height, width, channel = src_rgb.shape
    
    coltra_rgb = src_rgb
    
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = coltra_rgb[i,j,k]
                x = ((x-s_mean[k])*(t_std[k]/s_std[k]))+t_mean[k]
                # round or +0.5
                x = round(x)
                
                # boundary check 超過邊界拉回在邊界上
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                
                coltra_rgb[i,j,k] = x
    
    # 存顏色轉換結果圖
    coltra_bgr = cv2.cvtColor(coltra_rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite('coltra/0'+ str(coltra_num)+ '_' + coltra_name + '.png', coltra_bgr)
    
    # 畫結果值方圖
    show_histogram('c', coltra_rgb, coltra_name)
    
    return coltra_rgb, s_mean, s_std, t_mean, t_std


# In[4]:


def color_reverse(coltra_rgb, s_mean, s_std, t_mean, t_std, src_name, rev_num):
    
    height, width, channel = coltra_rgb.shape
    rev_rgb = coltra_rgb
    
    for i in range(0,height):
        for j in range(0,width):
            for k in range(0,channel):
                x = coltra_rgb[i,j,k]
                x = ((x-t_mean[k])*(s_std[k]/t_std[k]))+s_mean[k]
                # 無條件進位
                x = np.ceil(x)
                
                # boundary check 超過邊界拉回在邊界上
                x = 0 if x < 0 else x
                x = 255 if x > 255 else x
                
                rev_rgb[i,j,k] = x
    
    # 存顏色轉換結果圖
    rev_bgr = cv2.cvtColor(rev_rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite('revfct/0'+ str(rev_num)+ '_' + src_name + '_revfct.png', rev_bgr)
    
    return rev_rgb
    


# In[5]:


def output_mean_and_std_txt(coltra_name, s_mean, s_std, t_mean, t_std, coltra_num):
    path = 'sideinfodeci/0' + str(coltra_num)+ '_' + coltra_name + '.txt'
    f = open(path, 'w')
    
    f.write(str('{:.2f}'.format(s_mean[0]))+'\n')
    f.write(str('{:.2f}'.format(s_mean[1]))+'\n')
    f.write(str('{:.2f}'.format(s_mean[2]))+'\n')
    f.write(str('{:.2f}'.format(s_std[0]))+'\n')
    f.write(str('{:.2f}'.format(s_std[1]))+'\n')
    f.write(str('{:.2f}'.format(s_std[2]))+'\n')
    
    f.write(str('{:.2f}'.format(t_mean[0]))+'\n')
    f.write(str('{:.2f}'.format(t_mean[1]))+'\n')
    f.write(str('{:.2f}'.format(t_mean[2]))+'\n')
    f.write(str('{:.2f}'.format(t_std[0]))+'\n')
    f.write(str('{:.2f}'.format(t_std[1]))+'\n')
    f.write(str('{:.2f}'.format(t_std[2]))+'\n')

    f.close()


# In[6]:


def show_histogram(typeStr, img_rgb, img_name):

    plt.figure(figsize=(20,10))

    # 印出圖片
    plt.subplot(231)
    plt.imshow(img_rgb)
    if(typeStr == 's'):
        plt.title("source Name : " + img_name)
    elif(typeStr == 't'):
        plt.title("target Name : " + img_name)
    elif(typeStr == 'c'):
        plt.title("coltra Name : " + img_name)

    # 繪 RGB 三通道
    color = ('r', 'g', 'b')
    for i, col in enumerate(color):

        plt.subplot(232)
        # cv2.calcHist(影像, 通道, 遮罩, 區間數量, 數值範圍)
        hist = cv2.calcHist([img_rgb], [i], None, [256], [0, 256])
        plt.plot(hist, color=col)
        plt.title("RGB channel histogram")
        plt.xlabel("RBB value")
        plt.ylabel("frequent")
        plt.xlim([0, 256])

    hist_R = cv2.calcHist([img_rgb], [0], None, [256], [0, 256])
    hist_G = cv2.calcHist([img_rgb], [1], None, [256], [0, 256])
    hist_B = cv2.calcHist([img_rgb], [2], None, [256], [0, 256])
     
    #繪個別通道值方圖
    # red channel
    plt.subplot(234)
    plt.bar(range(1,257), hist_R.ravel(), color='r')
    plt.title("red channel histogram")
    plt.xlabel("R value")
    plt.ylabel("frequent")

    # green channel
    plt.subplot(235)
    plt.bar(range(1,257), hist_G.ravel(), color='g')
    plt.title("green channel histogram")
    plt.xlabel("G value")
    plt.ylabel("frequent")

    plt.subplot(236)
    plt.bar(range(1,257), hist_B.ravel(), color='b')
    plt.title("blue channel histogram")
    plt.xlabel("B value")
    plt.ylabel("frequent")
    


# In[7]:


def output_MSE_PSNR_SSIM(rev_rgb, src_rgb, src_name, img_num):
    
    img1 = rev_rgb.reshape(-1,3)
    img2 = s_rgb.reshape(-1,3)
    
    r_img1 = img1[:,0]
    r_img2 = img2[:,0]
    
    g_img1 = img1[:,1]
    g_img2 = img2[:,1]
    
    b_img1 = img1[:,2]
    b_img2 = img2[:,2]
    
    r_mse = np.around(mean_squared_error(r_img1, r_img2),2)
    g_mse = np.around(mean_squared_error(g_img1, g_img2),2)
    b_mse = np.around(mean_squared_error(b_img1, b_img2),2)
    r_psnr = np.around(peak_signal_noise_ratio(r_img1, r_img2),2)
    g_psnr = np.around(peak_signal_noise_ratio(g_img1, g_img2),2)
    b_psnr = np.around(peak_signal_noise_ratio(b_img1, b_img2),2)
    
    r_ssim = np.around(structural_similarity(r_img1, r_img2, data_range=r_img1.max() - r_img1.min()),6)
    g_ssim = np.around(structural_similarity(g_img1, g_img2, data_range=g_img1.max() - g_img1.min()),6)
    b_ssim = np.around(structural_similarity(b_img1, b_img2, data_range=b_img1.max() - b_img1.min()),6)
    
    path = 'revfct/0' + str(img_num)+ '_' + src_name + '.txt'
    f = open(path, 'w')
    
    lines_mse = [str('{:.2f}'.format(r_mse)), str(' {:.2f}'.format(g_mse)), str(' {:.2f}'.format(b_mse))+'\n']
    f.writelines(lines_mse)
    
    lines_psnr = [str('{:.2f}'.format(r_psnr)), str(' {:.2f}'.format(g_psnr)), str(' {:.2f}'.format(b_psnr))+'\n']
    f.writelines(lines_psnr)
    
    lines_ssim = [str('{:.6f}'.format(r_ssim)), str(' {:.6f}'.format(g_ssim)), str(' {:.6f}'.format(b_ssim))+'\n']
    f.writelines(lines_ssim)
    
    f.close()
    
    print("\n")
    print("rgb_mse: ",r_mse," ", g_mse," ", b_mse)
    print("rgb_psnr: ",r_psnr," ", g_psnr," ", b_psnr)
    print("rgb_ssim: ",r_ssim," ", g_ssim," ", b_ssim)


# In[8]:


global src_rgb

# 印出所有圖片
DATASRC = 'source/'
DATATRG = 'target/'

# 抓目錄下所有圖檔檔名
src_name = os.listdir(DATASRC)
tar_name = os.listdir(DATATRG)

for i in range (len(src_name)):
        
    # 圖片完整路徑
    src_path = DATASRC + src_name[i]
    tar_path = DATATRG + tar_name[i]
    
    # convert img to array 以彩色格式讀取(三維)
    s_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)
    src_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)   
    tar_bgr = cv2.imread(tar_path ,cv2.IMREAD_COLOR)
    
    # 原為 BGR 轉為 RGB
    s_rgb = cv2.cvtColor(s_bgr,cv2.COLOR_BGR2RGB)
    src_rgb = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2RGB)
    tar_rgb = cv2.cvtColor(tar_bgr,cv2.COLOR_BGR2RGB)
    
    # 畫值方圖
    show_histogram('s', src_rgb, src_name[i])
    show_histogram('t', tar_rgb, tar_name[i])
    
    # 去掉副檔名
    src_name_temp= Path(src_path).stem 
    tar_name_temp= Path(tar_path).stem 
    tar_name_tempp = tar_name_temp.lstrip('0'+str(i+1))
    coltra_name = src_name_temp +'_'+ tar_name_tempp
    
    # 做色彩轉換
    coltra_rgb, s_mean, s_std, t_mean, t_std = color_transfer(src_rgb, tar_rgb, src_name_temp, tar_name_temp, coltra_name, i+1)
    
    # 做色彩反轉
    rev_rgb = color_reverse(coltra_rgb, s_mean, s_std, t_mean, t_std, src_name_temp, i+1)
    
    # 量化 逆向色彩轉移影像 與 原始來源影像之 MSE、PSNR、SSIM
    output_MSE_PSNR_SSIM(rev_rgb, s_rgb, src_name_temp, i+1)

