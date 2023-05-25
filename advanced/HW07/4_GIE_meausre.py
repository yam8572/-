import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名
import csv

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
    csv_file = 'statis/GIE_res.csv'
    data = [
    ['GIE',' ','Plain',' ',' ','Cipher',' ',' '],
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
            total_pixels = M * N

            # Split the plain image and the encrypted image into RGB channels
            plain_channels = cv2.split(plain_rgb)
            cipher_channels = cv2.split(cipher_rgb)

            # Initialize variables to store histograms and entropies
            plain_histograms = []
            cipher_histograms = []
            plain_global_entropies = []
            cipher_global_entropies = []

            # Calculate the histogram for each channel of the plain image
            for channel in plain_channels:
                plain_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                plain_histograms.append(plain_hist)

            # Calculate the histogram for each channel of the encrypted image
            for channel in cipher_channels:
                cipher_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
                cipher_histograms.append(cipher_hist)

            # Normalize the histograms for both images
            plain_probability_histograms = [hist / total_pixels for hist in plain_histograms]
            cipher_probability_histograms = [hist / total_pixels for hist in cipher_histograms]

            # Calculate the entropy for each channel of the plain image
            # 1e-10 避免 Log2(0) 錯誤
            for plain_probability_s in plain_probability_histograms:
                # probability_of_s = plain_normalized_hist / 256
                plain_entropy = -np.sum(plain_probability_s * np.log2(plain_probability_s + 1e-10))
                plain_global_entropies.append(plain_entropy)

            # Calculate the entropy for each channel of the encrypted image
            for cipher_probability_s in cipher_probability_histograms:
                cipher_entropy = -np.sum(cipher_probability_s * np.log2(cipher_probability_s + 1e-10))
                cipher_global_entropies.append(cipher_entropy)

            print("plain_global_entropies RGB ",plain_global_entropies)
            print("cipher_global_entropies RGB ",cipher_global_entropies)
            print()

            if(K==8):
                writer.writerow([plain_name_temp,'gray', str(plain_global_entropies[0]),'','', str(cipher_global_entropies[0])])
            else: # K=24
                writer.writerow([plain_name_temp,'color', str(plain_global_entropies[0]) ,str(plain_global_entropies[1]),str(plain_global_entropies[2]), str(cipher_global_entropies[0]),str(cipher_global_entropies[1]),str(cipher_global_entropies[2])])