import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名
import csv
import random

def _1Dto2D(val:int):
    row = int((val - (val % M)) / M)
    col = int(val % M)
    return row, col

def _2Dto1D(row:int, col:int):
    s = row * M + col
    return s

def calPearson(x,y):

    # Calculate the mean of x and y
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)

    # Calculate the covariance and variances
    covariance = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y))
    variance_x = sum((xi - mean_x) ** 2 for xi in x)
    variance_y = sum((yi - mean_y) ** 2 for yi in y)

    # Calculate the Pearson correlation coefficient
    correlation = covariance / (variance_x ** 0.5 * variance_y ** 0.5)

    return correlation


def check(val:int, lower_bound:int, upper_bound:int):
    _pass = False 
    r,c = _1Dto2D(val)
    horizontal_r = r
    horizontal_c = c+1
    vertical_r = r+1
    vertical_c = c
    diagonal_r = r+1
    diagonal_c = c+1
    
    horizontal_s = _2Dto1D(horizontal_r,horizontal_c)
    vertical_s = _2Dto1D(vertical_r,vertical_c)
    diagonal_s = _2Dto1D(diagonal_r,diagonal_c)

    if (lower_bound <= horizontal_s <= upper_bound) and (lower_bound <= vertical_s <= upper_bound) and (lower_bound <= diagonal_s <= upper_bound):
        _pass = True

    return _pass

if __name__ == '__main__':
    DATASRC = 'source/'
    DATATRG = 'encrypt/'
    # Set the number of bins for the histogram
    bins = 256
    # 抓原始圖片目錄下所有圖檔檔名
    plain_name = os.listdir(DATASRC)
    # 抓加密圖片目錄下所有圖檔檔名
    cipher_name = os.listdir(DATATRG)
    # Open the CSV file in write mode
    csv_file = 'statis/COR_res.csv'
    data = [
    ['COR',' ','Plain',' ',' ',' ',' ',' ',' ',' ',' ','Cipher',' ',' '],
    ['Sample','8000','red','','','green','','','blue','','','red','','','green','','','blue','',''],
    ['Image','Type','horizontal','vertical','diagonal','horizontal','vertical','diagonal','horizontal','vertical ','diagonal','horizontal','vertical','diagonal','horizontal','vertical','diagonal','horizontal','vertical','diagonal']
    ]   
    with open(csv_file, 'w', newline='') as file:
        # Create a CSV writer object
        writer = csv.writer(file)
        writer.writerows(data)

        for i in range (len(plain_name)):

            # 原始圖片完整路徑
            plain_path = DATASRC + plain_name[i]
            print(plain_path)

            # 去掉副檔名
            plain_name_temp= Path(plain_path).stem

            # 加密圖片完整路徑
            cipher_path = DATATRG + cipher_name[i]
            print(cipher_path)

            # Load the image
            plain_bgr = cv2.imread(plain_path, cv2.IMREAD_COLOR)
            cipher_bgr = cv2.imread(cipher_path, cv2.IMREAD_COLOR)
            # Convert the image to grayscale
            plain_gray = cv2.cvtColor(plain_bgr, cv2.COLOR_BGR2GRAY)
            sub1 = plain_bgr[:, :, 0] - plain_gray
            sub2 = plain_bgr[:, :, 1] - plain_gray
            sub3 = plain_bgr[:, :, 2] - plain_gray
            #判斷sub中的每个值是不是都等於0
            if((sub1 == 0).all() and (sub2 == 0).all() and (sub3 == 0).all()):
                type=8
                print("The image is grayscale")
            else:
                type=24
                print("The image is color")
                
            # 原為 BGR 轉為 RGB
            plain_rgb = cv2.cvtColor(plain_bgr,cv2.COLOR_BGR2RGB)
            cipher_rgb = cv2.cvtColor(cipher_bgr,cv2.COLOR_BGR2RGB)

            # N 為影像之水平像素數量，M 為影像之垂直像素數量
            M, N, C = plain_rgb.shape
            L = M * N
            K = 8000
            l=np.floor(L/K)
            # 隨機選取 >> 每距離 l 取一點
            selected_values = []
            for i in range(0, L, int(l)):
                # ex: random.randint(0,16) >> random.randint(1,15) 讓抽樣點不包含 0 和 16 >> 不取分隔點上的點
                value = random.randint(i+1, min(i+l-1, L-1))
                selected_values.append(value)

            # 因無法等分切割，會有多取超過8000點 >> 隨機 remove 掉 選擇點 直到=K 希望取的sample點個數
            while len(selected_values) > K:
                random_indices = random.sample(range(len(selected_values)), k=1)
                del selected_values[random_indices[0]]

            # 1D(s)轉成2D(r,c) 並 分別對應圖片像素點值在 rgb channel 上
            plain_x_r=[]
            plain_horizontal_y_r=[]
            plain_vertical_y_r=[]
            plain_diagonal_y_r=[]
            plain_x_g=[]
            plain_horizontal_y_g=[]
            plain_vertical_y_g=[]
            plain_diagonal_y_g=[]
            plain_x_b=[]
            plain_horizontal_y_b=[]
            plain_vertical_y_b=[]
            plain_diagonal_y_b=[]

            cipher_x_r=[]
            cipher_horizontal_y_r=[]
            cipher_vertical_y_r=[]
            cipher_diagonal_y_r=[]
            cipher_x_g=[]
            cipher_horizontal_y_g=[]
            cipher_vertical_y_g=[]
            cipher_diagonal_y_g=[]
            cipher_x_b=[]
            cipher_horizontal_y_b=[]
            cipher_vertical_y_b=[]
            cipher_diagonal_y_b=[]

            for item in selected_values:
                r,c = _1Dto2D(item)
                # print(f"r={r} c={c}")
                plain_x_r.append(plain_rgb[c,r,0])
                plain_x_g.append(plain_rgb[c,r,1])
                plain_x_b.append(plain_rgb[c,r,2])

                plain_horizontal_y_r.append(plain_rgb[min(c+1,M-1),r,0])
                plain_horizontal_y_g.append(plain_rgb[min(c+1,M-1),r,1])
                plain_horizontal_y_b.append(plain_rgb[min(c+1,M-1),r,2])

                plain_vertical_y_r.append(plain_rgb[c,min(r+1,N-1),0])
                plain_vertical_y_g.append(plain_rgb[c,min(r+1,N-1),1])
                plain_vertical_y_b.append(plain_rgb[c,min(r+1,N-1),2])

                plain_diagonal_y_r.append(plain_rgb[min(c+1,M-1),min(r+1,N-1),0])
                plain_diagonal_y_g.append(plain_rgb[min(c+1,M-1),min(r+1,N-1),1])
                plain_diagonal_y_b.append(plain_rgb[min(c+1,M-1),min(r+1,N-1),2])

                cipher_x_r.append(cipher_rgb[c,r,0])
                cipher_x_g.append(cipher_rgb[c,r,1])
                cipher_x_b.append(cipher_rgb[c,r,2])

                cipher_horizontal_y_r.append(cipher_rgb[min(c+1,M-1),r,0])
                cipher_horizontal_y_g.append(cipher_rgb[min(c+1,M-1),r,1])
                cipher_horizontal_y_b.append(cipher_rgb[min(c+1,M-1),r,2])

                cipher_vertical_y_r.append(cipher_rgb[c,min(r+1,N-1),0])
                cipher_vertical_y_g.append(cipher_rgb[c,min(r+1,N-1),1])
                cipher_vertical_y_b.append(cipher_rgb[c,min(r+1,N-1),2])

                cipher_diagonal_y_r.append(cipher_rgb[min(c+1,M-1),min(r+1,N-1),0])
                cipher_diagonal_y_g.append(cipher_rgb[min(c+1,M-1),min(r+1,N-1),1])
                cipher_diagonal_y_b.append(cipher_rgb[min(c+1,M-1),min(r+1,N-1),2])

            plain_corr_hor_r=calPearson(plain_x_r,plain_horizontal_y_r)
            plain_corr_ver_r=calPearson(plain_x_r,plain_vertical_y_r)
            plain_corr_diag_r=calPearson(plain_x_r,plain_diagonal_y_r)

            plain_corr_hor_g=calPearson(plain_x_g,plain_horizontal_y_g)
            plain_corr_ver_g=calPearson(plain_x_g,plain_vertical_y_g)
            plain_corr_diag_g=calPearson(plain_x_g,plain_diagonal_y_g)

            plain_corr_hor_b=calPearson(plain_x_b,plain_horizontal_y_b)
            plain_corr_ver_b=calPearson(plain_x_b,plain_vertical_y_b)
            plain_corr_diag_b=calPearson(plain_x_b,plain_diagonal_y_b)

            cipher_corr_hor_r=calPearson(cipher_x_r,cipher_horizontal_y_r)
            cipher_corr_ver_r=calPearson(cipher_x_r,cipher_vertical_y_r)
            cipher_corr_diag_r=calPearson(cipher_x_r,cipher_diagonal_y_r)

            cipher_corr_hor_g=calPearson(cipher_x_g,cipher_horizontal_y_g)
            cipher_corr_ver_g=calPearson(cipher_x_g,cipher_vertical_y_g)
            cipher_corr_diag_g=calPearson(cipher_x_g,cipher_diagonal_y_g)

            cipher_corr_hor_b=calPearson(cipher_x_b,cipher_horizontal_y_b)
            cipher_corr_ver_b=calPearson(cipher_x_b,cipher_vertical_y_b)
            cipher_corr_diag_b=calPearson(cipher_x_b,cipher_diagonal_y_b)
            if(type==8):
                writer.writerow([plain_name_temp,'gray', str(plain_corr_hor_r), str(plain_corr_ver_r),str(plain_corr_diag_r),
                                 '','','','','','',
                                 str(cipher_corr_hor_r),str(cipher_corr_ver_r),str(cipher_corr_diag_r)])
            else: # type=24
                writer.writerow([plain_name_temp,'color', str(plain_corr_hor_r) ,str(plain_corr_ver_r),str(plain_corr_diag_r), str(plain_corr_hor_g) ,str(plain_corr_ver_g),str(plain_corr_diag_g),str(plain_corr_hor_b) ,str(plain_corr_ver_b),str(plain_corr_diag_b),
                                 str(cipher_corr_hor_r) ,str(cipher_corr_ver_r),str(cipher_corr_diag_r), str(cipher_corr_hor_g) ,str(cipher_corr_ver_g),str(cipher_corr_diag_g),str(cipher_corr_hor_b) ,str(cipher_corr_ver_b),str(cipher_corr_diag_b)])
            print()