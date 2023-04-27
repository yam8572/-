import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名

# data visualisation and manipulation
import matplotlib.pyplot as plt #show img and table

from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity

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

def color_transfer(src_rgb, tar_rgb, src_name, tar_name, coltra_name, coltra_num, weight_r, weight_g, weight_b):
    
    s_mean, s_std = get_mean_and_std('s', src_name, src_rgb)
    t_mean, t_std = get_mean_and_std('t', tar_name, tar_rgb)
    
    coltra_rgb = src_rgb

    for k in range(0,3):
        x = src_rgb[:,:,k]
        if(k==0):
            # red channel
            x = ((x-s_mean[k])*(((weight_r*t_std[k])+(1-weight_r)*s_std[k])/s_std[k]))+ weight_r*t_mean[k]+(1-weight_r)*s_mean[k]
        elif(k==1):
            # green channel
            x = ((x-s_mean[k])*(((weight_g*t_std[k])+(1-weight_g)*s_std[k])/s_std[k]))+ weight_g*t_mean[k]+(1-weight_g)*s_mean[k]
        elif(k==2):
            # blue channel
            x = ((x-s_mean[k])*(((weight_b*t_std[k])+(1-weight_b)*s_std[k])/s_std[k]))+ weight_b*t_mean[k]+(1-weight_b)*s_mean[k]

        # boundary check 超過邊界拉回在邊界上
        x = np.clip(x, 0, 255)
        coltra_rgb[:,:,k] = x
    
    # 存顏色轉換結果圖
    str_weight = '_' + str(weight_r) + '_' + str(weight_g) + '_' + str(weight_b)
    coltra_bgr = cv2.cvtColor(coltra_rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite('result/'+ coltra_name + str_weight + '.png', coltra_bgr)
    
    return coltra_rgb, s_mean, s_std, t_mean, t_std

if __name__ == '__main__':

    # weight_r, weight_g, weight_b = input("Enter 3 channel weight_red , weight_green, weight_blue(0.0 ≤ w ≤ 1.0):").split()
    # weight_r = float(weight_r)
    # weight_g = float(weight_g)
    # weight_b = float(weight_b)

    global src_rgb

    # 印出所有圖片
    DATASRC = 'source/'
    DATATRG = 'encrypt/'

    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)

    for i in range (len(src_name)):
            
        # 圖片完整路徑
        src_path = DATASRC + src_name[i]
        src_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)   
        print(src_bgr.type())
        
        # 原為 BGR 轉為 RGB
        src_rgb = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2RGB)
        
        # 去掉副檔名
        src_name_temp= Path(src_path).stem
        

