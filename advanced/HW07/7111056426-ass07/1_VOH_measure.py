import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名
import csv

def calculate_voh(image, bins):
    hist_R = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_G = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_B = cv2.calcHist([image], [2], None, [bins], [0, 256])

    voh_R = np.var(hist_R)
    voh_G = np.var(hist_G)
    voh_B = np.var(hist_B)

    return voh_R, voh_G, voh_B

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
    csv_file = 'statis/VOH_res.csv'
    data = [
    ['VOH',' ','Plain',' ',' ','Cipher',' ',' '],
    ['Image','Type','Red','Green','Blue','Red','Green','Blue'],
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
                K=8
                print("The image is grayscale")
            else:
                K=24
                print("The image is color")
                
            # 原為 BGR 轉為 RGB
            plain_rgb = cv2.cvtColor(plain_bgr,cv2.COLOR_BGR2RGB)
            cipher_rgb = cv2.cvtColor(cipher_bgr,cv2.COLOR_BGR2RGB)

            # N 為影像之水平像素數量，M 為影像之垂直像素數量
            M, N, C = plain_rgb.shape

            # Calculate the VOH for each channel
            plain_voh_r, plain_voh_g, plain_voh_b = calculate_voh(plain_rgb, bins)
            print("plain VOH for R channel:", plain_voh_r)
            if(K==24):
                print("plain VOH for G channel:", plain_voh_g)
                print("plain VOH for B channel:", plain_voh_b)

            # Calculate the VOH for each channel
            cipher_voh_r, cipher_voh_g, cipher_voh_b = calculate_voh(cipher_rgb, bins)
            print("cipher VOH for R channel:", cipher_voh_r)
            if(K==24):
                print("cipher VOH for G channel:", cipher_voh_g)
                print("cipher VOH for B channel:", cipher_voh_b)
            print()

            if(K==8):
                writer.writerow([plain_name_temp,'gray', str(plain_voh_r),'','', str(cipher_voh_r)])
            else: # K=24
                writer.writerow([plain_name_temp,'color', str(plain_voh_r) ,str(plain_voh_g),str(plain_voh_b), str(cipher_voh_r),str(plain_voh_g),str(plain_voh_b)])
