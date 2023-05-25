import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path
from scipy.stats import chisquare
from pathlib import Path # 去掉副檔名
import csv

# Function to calculate Chi-square statistic and p-value
def chi_square_test(cipher_hist, M, N):
    # Calculate the observed and expected frequencies
    observed = np.array(cipher_hist)
    expected = M*N / 256

    # Calculate the Chi-square statistic
    chi2 = np.sum((observed - expected)**2 / expected)

    return chi2

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
    csv_file = 'statis/CHI_res.csv'
    data = [
    ['CHI',' ','Cipher',' ',' ',' ',' ',' ','Results'],
    ['Image','Type','Red','Green','alpha','chi value', 'Red','Green','Blue']
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
            # 判斷sub中的每个值是不是都等於0
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

            cipher_hist_R = cv2.calcHist([cipher_rgb], [0], None, [bins], [0, 256])
            cipher_hist_G = cv2.calcHist([cipher_rgb], [1], None, [bins], [0, 256])
            cipher_hist_B = cv2.calcHist([cipher_rgb], [2], None, [bins], [0, 256])

            # Calculate Chi-square statistics
            cipher_chi2_R = chi_square_test(cipher_hist_R, M, N)
            cipher_chi2_G = chi_square_test(cipher_hist_G, M, N)
            cipher_chi2_B = chi_square_test(cipher_hist_B, M, N)

            print("cipher for R channel:", cipher_chi2_R)
            if(K==24):
                print("cipher for G channel:", cipher_chi2_G)
                print("cipher for B channel:", cipher_chi2_B)

            threshold = 293.248
            alpha = 0.05
            if cipher_chi2_R < threshold:
                result_R = 'Pass'
                print(f"Red channel Pass: encrypted image is uniformly distributed")
            else:
                result_R = 'Fail'
                print(f"Red channel Fail: encrypted image is not uniformly distributed")
            
            if(K==24):
                if cipher_chi2_G < threshold:
                    result_G = 'Pass'
                    print(f"Green channel Pass: encrypted image is uniformly distributed")
                else:
                    result_G = 'Fail'
                    print(f"Green channel Fail: encrypted image is not uniformly distributed")

                if cipher_chi2_B < threshold:
                    result_B = 'Pass'
                    print(f"Blue channel Pass: encrypted image is uniformly distributed")
                else:
                    result_B = 'Fail'
                    print(f"Blue channel Fail: encrypted image is not uniformly distributed")

            print()

            if(K==8):
                writer.writerow([plain_name_temp,'gray', str(cipher_chi2_R),'','', str(alpha), str(threshold), result_R])
            else: # K=24
                writer.writerow([plain_name_temp,'color', str(cipher_chi2_R) ,str(cipher_chi2_G),str(cipher_chi2_B), str(alpha), str(threshold), result_R, result_G, result_B])
