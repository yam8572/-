import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名
from scipy.spatial import distance as dist

def get_mean_and_std(x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    
    return x_mean, x_std

def color_transfer(src_rgb, tar_rgb, weight_r, weight_g, weight_b):
    
    global s_mean, s_std, t_mean, t_std
    coltra_rgb = np.ndarray(src_rgb.shape, dtype=np.uint8)

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

        # # round or +0.5
        x = np.round(x)
        
        # boundary check 超過邊界拉回在邊界上
        x = np.clip(x, 0, 255)
        coltra_rgb[:,:,k] = x
    
    return coltra_rgb

# Brute Force Method:
# 101 種可能
def find_optimal_weight(src_rgb, tar_rgb, src_hist, tar_hist, i):
    
    global method
    opt_wr=0
    opt_wg=0
    opt_wb=0
    fix_w=0.5
    best_dis_r=1000
    best_dis_g=1000
    best_dis_b=1000

    for w_r in np.arange(0, 1.01, 0.01):

        # 做色彩轉換
        coltra_rgb_r = color_transfer(src_rgb, tar_rgb, w_r, fix_w, fix_w)
        coltra_hist_r = cv2.calcHist([coltra_rgb_r], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        coltra_hist_r = cv2.normalize(coltra_hist_r, coltra_hist_r).flatten()

        # 計算 distance 比較 coltra_rgb 和 src_rgb & tar_rgb 的距離
        src_d_r = cv2.compareHist(src_hist, coltra_hist_r, method)
        tar_d_r = cv2.compareHist(tar_hist, coltra_hist_r, method)	
        dis_r = np.abs(src_d_r - tar_d_r)

        # update optimal
        if(best_dis_r > dis_r):
            best_dis_r = dis_r
            opt_wr = w_r

    for w_g in np.arange(0, 1.01, 0.01):

        # 做色彩轉換
        coltra_rgb_g = color_transfer(src_rgb, tar_rgb, opt_wr, w_g, fix_w)
        coltra_hist_g = cv2.calcHist([coltra_rgb_g], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        coltra_hist_g = cv2.normalize(coltra_hist_g, coltra_hist_g).flatten()

        # 計算 distance 比較 coltra_rgb 和 src_rgb & tar_rgb 的距離
        src_d_g = cv2.compareHist(src_hist, coltra_hist_g, method)
        tar_d_g = cv2.compareHist(tar_hist, coltra_hist_g, method)	
        dis_g = np.abs(src_d_g - tar_d_g)

        # update optimal
        if(best_dis_g > dis_g):
            best_dis_g = dis_g
            opt_wg = w_g

    for w_b in np.arange(0, 1.01, 0.01):

        # 做色彩轉換
        coltra_rgb_b = color_transfer(src_rgb, tar_rgb, opt_wr, opt_wg, w_b)
        coltra_hist_b = cv2.calcHist([coltra_rgb_b], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        coltra_hist_b = cv2.normalize(coltra_hist_b, coltra_hist_b).flatten()

        # 計算 distance 比較 coltra_rgb 和 src_rgb & tar_rgb 的距離
        src_d_b = cv2.compareHist(src_hist, coltra_hist_b, method)
        tar_d_b = cv2.compareHist(tar_hist, coltra_hist_b, method)	
        dis_b = np.abs(src_d_b - tar_d_b)

        # update optimal
        if(best_dis_b > dis_b):
            best_dis_b = dis_b
            opt_wb = w_b

    # 存最佳權重顏色轉換結果圖
    opt_wr = np.around(opt_wr,2)
    opt_wg = np.around(opt_wg,2)
    opt_wb = np.around(opt_wb,2)
    str_weight = str(opt_wr) + '_' + str(opt_wg) + '_' + str(np.around(opt_wb,2))
    print("stop weight:",str_weight)
    coltra_rgb = color_transfer(src_rgb, tar_rgb, opt_wr, opt_wg, opt_wb)
    coltra_bgr = cv2.cvtColor(coltra_rgb,cv2.COLOR_RGB2BGR)
    cv2.imwrite('result_bru/WCT0'+ str(i) +'-Bru-'+ str_weight + '.png', coltra_bgr)
    
    return opt_wr, opt_wg, opt_wb

if __name__ == '__main__':

    OPENCV_METHODS = (
    ("Correlation", cv2.HISTCMP_CORREL),
    ("Chi-Squared", cv2.HISTCMP_CHISQR),
    ("Intersection", cv2.HISTCMP_INTERSECT),
    ("Hellinger", cv2.HISTCMP_BHATTACHARYYA))
    
    method=input("choose the distance method: 1:Correlation 2:Chi-Squared 3.Intersection 4.Hellinger: ")
    method = int(method) - 1
    
    # 印出所有圖片
    DATASRC = 'source/'
    DATATRG = 'target/'
    DATAOPT = 'result_bru/'

    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)
    tar_name = os.listdir(DATATRG)
    opt_name = os.listdir(DATAOPT)

    s_mean=0
    s_std =0
    t_mean=0
    t_std=0

    for i in range (len(src_name)):
    # for i in range (1):
            
        # 圖片完整路徑
        src_path = DATASRC + src_name[i]
        tar_path = DATATRG + tar_name[i]
        
        # convert img to array 以彩色格式讀取(三維)
        src_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)   
        tar_bgr = cv2.imread(tar_path ,cv2.IMREAD_COLOR)
        
        # 原為 BGR 轉為 RGB
        src_rgb = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2RGB)
        tar_rgb = cv2.cvtColor(tar_bgr,cv2.COLOR_BGR2RGB)

        s_mean, s_std = get_mean_and_std(src_rgb)
        t_mean, t_std = get_mean_and_std(tar_rgb)

        # extract a 3D RGB color histogram from the image,
        # using 8 bins per channel, normalize, and update the hist_dict
        src_hist = cv2.calcHist([src_rgb], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        src_hist = cv2.normalize(src_hist, src_hist).flatten()

        tar_hist = cv2.calcHist([tar_rgb], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        tar_hist = cv2.normalize(tar_hist, tar_hist).flatten()

        find_optimal_weight(src_rgb, tar_rgb, src_hist, tar_hist, i+1)


    for i in range (len(opt_name)):

        # 圖片完整路徑
        src_path = DATASRC + src_name[i]
        tar_path = DATATRG + tar_name[i]
        opt_path = DATAOPT + opt_name[i]

        # convert img to array 以彩色格式讀取(三維)
        src_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)   
        tar_bgr = cv2.imread(tar_path ,cv2.IMREAD_COLOR)
        opt_bgr = cv2.imread(opt_path ,cv2.IMREAD_COLOR)
        print(opt_path)

        # 原為 BGR 轉為 RGB
        src_rgb = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2RGB)
        tar_rgb = cv2.cvtColor(tar_bgr,cv2.COLOR_BGR2RGB)
        opt_rgb = cv2.cvtColor(opt_bgr,cv2.COLOR_BGR2RGB)

        src_hist = cv2.calcHist([src_rgb], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        src_hist = cv2.normalize(src_hist, src_hist).flatten()

        tar_hist = cv2.calcHist([tar_rgb], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        tar_hist = cv2.normalize(tar_hist, tar_hist).flatten()

        opt_hist = cv2.calcHist([opt_rgb], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
        opt_hist = cv2.normalize(opt_hist, opt_hist).flatten()

        src_d = cv2.compareHist(src_hist, opt_hist, method)
        tar_d= cv2.compareHist(tar_hist, opt_hist, method)

        print("src_d=",src_d,"tar_d=",tar_d)
        print("difference=",np.abs(src_d-tar_d))
        print()