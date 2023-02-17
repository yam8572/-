import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名
# data visualisation and manipulation
import matplotlib.pyplot as plt #show img and table
import pandas as pd # 匯出 csv

def get_mean_and_std(typeStr, img_name, x):
    x_mean, x_std = cv2.meanStdDev(x)
    x_mean = np.hstack(np.around(x_mean,2))
    x_std = np.hstack(np.around(x_std,2))
    
    # 印出平均與標準差
    if(typeStr == 's'):
        print("\n來源圖檔名: ", img_name)
    elif(typeStr == 't'):
        print("\n目標圖檔名: ", img_name)

    MeanStd = [x_mean,x_std]
    MeanStd_table = pd.DataFrame(MeanStd)
    col_name = ['red','green','blue']
    row_name = ['Mean','Standard Deviation']
    MeanStd_table.columns = col_name
    MeanStd_table.index = row_name
    print(MeanStd_table)

    # 匯出 answer table 成 csv 檔
    MeanStd_table.to_csv('feature/'+ img_name + '_dec.csv')
    
    return x_mean, x_std

if __name__ == '__main__':

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
        src_bgr = cv2.imread(src_path ,cv2.IMREAD_COLOR)   
        tar_bgr = cv2.imread(tar_path ,cv2.IMREAD_COLOR)
        
        # 原為 BGR 轉為 RGB
        src_rgb = cv2.cvtColor(src_bgr,cv2.COLOR_BGR2RGB)
        tar_rgb = cv2.cvtColor(tar_bgr,cv2.COLOR_BGR2RGB)
        
        # 去掉副檔名
        src_name_temp= Path(src_path).stem 
        tar_name_temp= Path(tar_path).stem 
        
        # 匯出平均及標準差成csv
        s_mean, s_std = get_mean_and_std('s', src_name_temp, src_rgb)
        t_mean, t_std = get_mean_and_std('t', tar_name_temp, tar_rgb)

        

