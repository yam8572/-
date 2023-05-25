# CHI_res.csv
import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path
from scipy.stats import chisquare

# Function to calculate Chi-square statistic and p-value
def chi_square_test(hist1, hist2):
    # Calculate the observed and expected frequencies
    observed = np.array(hist1)
    expected = np.array(hist2) 

    # Calculate the Chi-square statistic
    chi2 = np.sum((observed - expected)**2 / expected)

    # Calculate the degrees of freedom
    degrees_of_freedom = len(hist1) - 1

    # Calculate the p-value using the Chi-square distribution
    p_value = 1.0 - chi2_distribution_cdf(chi2, degrees_of_freedom)

    return chi2, p_value

# Function to calculate the cumulative distribution function (CDF) of the Chi-square distribution
def chi2_distribution_cdf(chi2, df):
    # Calculate the incomplete gamma function
    gamma = lambda s, x: np.exp(-x) * np.power(x, s) / np.math.gamma(s + 1.0)
  
    # Calculate the CDF using the regularized gamma function
    cdf = lambda x, df: 1.0 - gamma(df / 2.0, x / 2.0) / np.math.gamma(df / 2.0)
  
    return cdf(chi2, df)

if __name__ == '__main__':
    DATASRC = 'source/'
    DATATRG = 'encrypt/'
    # Set the number of bins for the histogram
    bins = 256
    # 抓原始圖片目錄下所有圖檔檔名
    plain_name = os.listdir(DATASRC)
    # 抓加密圖片目錄下所有圖檔檔名
    cipher_name = os.listdir(DATATRG)
    for i in range (len(plain_name)):
        
        # 原始圖片完整路徑
        plain_path = DATASRC + plain_name[i]
        print(plain_path)

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

        # Split the images into RGB channels
        channels1 = cv2.split(plain_rgb)
        channels2 = cv2.split(cipher_rgb)

        # Initialize lists to store the Chi-square statistics and p-values for each channel
        chi2_values = []
        p_values = []

        # Iterate over the channels and perform the Chi-square test
        for i in range(3):  # 3 for RGB channels
            # Calculate the histograms for the current channel
            hist1, _ = np.histogram(channels1[i].flatten(), bins=256, range=[0, 256])
            hist2, _ = np.histogram(channels2[i].flatten(), bins=256, range=[0, 256])

            # Convert histograms to lists or numpy arrays
            hist1 = hist1.tolist()
            hist2 = hist2.tolist()

            # Perform the Chi-square test for the current channel
            chi2, p = chisquare(hist1, hist2)

            # Store the results for the current channel
            chi2_values.append(chi2)
            p_values.append(p)

        print("cipher for R channel:", chi2_values[0])
        print("cipher for G channel:", chi2_values[1])
        print("cipher for B channel:", chi2_values[2])

        print("p_values for R channel:", p_values[0])
        print("p_values for G channel:", p_values[1])
        print("p_values for B channel:", p_values[2])

        # Compare p-values with significance level
        alpha = 0.05
        for i in range(3):  # 3 for RGB channels
            if p_values[i] < alpha:
                print(f"For channel {i+1}: Rejec(H1):histograms is not uniform distribution. There is a significant difference in image histograms.")
            else:
                print(f"For channel {i+1}: Accept (H0):histograms is uniform distribution. There is no significant difference in image histograms.")

        

        # cv2.calcHist(images, channels, mask, histSize, ranges)
        # plain_hist_r = cv2.calcHist([plain_rgb],[0],None,[256],[0,256], accumulate = False)
        # plain_hist_g = cv2.calcHist([plain_rgb],[1],None,[256],[0,256], accumulate = False)
        # plain_hist_b = cv2.calcHist([plain_rgb],[2],None,[256],[0,256], accumulate = False)

        # cipher_hist_r = cv2.calcHist([cipher_rgb],[0],None,[256],[0,256], accumulate = False)
        # cipher_hist_g = cv2.calcHist([cipher_rgb],[1],None,[256],[0,256], accumulate = False)
        # cipher_hist_b = cv2.calcHist([cipher_rgb],[2],None,[256],[0,256], accumulate = False)

        # res_r = cv2.compareHist(plain_hist_r, cipher_hist_r, cv2.HISTCMP_CHISQR)
        # res_g = cv2.compareHist(plain_hist_g, cipher_hist_g, cv2.HISTCMP_CHISQR)
        # res_b = cv2.compareHist(plain_hist_b, cipher_hist_b, cv2.HISTCMP_CHISQR)

        # print("cipher for R channel:", res_r)
        # print("cipher for G channel:", res_g)
        # print("cipher for B channel:", res_b)
