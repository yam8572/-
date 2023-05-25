# VOH_res.csv
import cv2 # opencv library read img operations
import numpy as np # array operations
import os # use directory & join path

def calculate_voh(image, bins):
    voh_channels = []
    
    for channel in range(image.shape[2]):
        # Extract the channel from the image
        channel_image = image[:,:,channel]
        
        # Calculate the histogram
        hist = cv2.calcHist([channel_image], [0], None, [bins], [0, 256])

        # Calculate the variance of the histogram
        variance = np.var(hist)
        
        # # Calculate the mean of the histogram
        # mean = np.mean(hist)
        
        # # Calculate the squared differences from the mean
        # squared_diffs = (hist - mean) ** 2
        
        # # Calculate the variance of the histogram
        # voh = np.mean(squared_diffs)
        
        voh_channels.append(variance)
    
    return voh_channels




if __name__ == '__main__':
    DATASRC = 'source/'
    DATATRG = 'encrypt/'
    # Set the number of bins for the histogram
    bins = 256
    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)
    for i in range (len(src_name)):
        
        # 圖片完整路徑
        src_path = DATASRC + src_name[i]
        print(src_path)

        # Load the image
        src_decimal_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        # Convert the image to grayscale
        src_dec_gray = cv2.cvtColor(src_decimal_bgr, cv2.COLOR_BGR2GRAY)
        sub1 = src_decimal_bgr[:, :, 0] - src_dec_gray
        sub2 = src_decimal_bgr[:, :, 1] - src_dec_gray
        sub3 = src_decimal_bgr[:, :, 2] - src_dec_gray
        #判斷sub中的每个值是不是都等於0
        if((sub1 == 0).all() and (sub2 == 0).all() and (sub3 == 0).all()):
            K=8
            print("The image is grayscale")
        else:
            K=24
            print("The image is color")
        
        # 原為 BGR 轉為 RGB
        src_decimal_rgb = cv2.cvtColor(src_decimal_bgr,cv2.COLOR_BGR2RGB)

        # Calculate the VOH for each channel
        voh_channels = calculate_voh(src_decimal_rgb, bins)
        print("VOH for R channel:", voh_channels[0])
        print("VOH for G channel:", voh_channels[1])
        print("VOH for B channel:", voh_channels[2])

import cv2
import numpy as np

# Load the plain image and the encrypted image
plain_image = cv2.imread("plain_image.jpg")
encrypted_image = cv2.imread("encrypted_image.jpg")

# Split the plain image and the encrypted image into RGB channels
plain_channels = cv2.split(plain_image)
encrypted_channels = cv2.split(encrypted_image)

# Initialize variables to store histograms and entropies
plain_histograms = []
encrypted_histograms = []
plain_entropies = []
encrypted_entropies = []

# Calculate the histogram for each channel of the plain image
for channel in plain_channels:
    plain_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    plain_histograms.append(plain_hist)

# Calculate the histogram for each channel of the encrypted image
for channel in encrypted_channels:
    encrypted_hist = cv2.calcHist([channel], [0], None, [256], [0, 256])
    encrypted_histograms.append(encrypted_hist)

# Normalize the histograms for both images
plain_total_pixels = plain_image.shape[0] * plain_image.shape[1]
plain_normalized_histograms = [hist / plain_total_pixels for hist in plain_histograms]

encrypted_total_pixels = encrypted_image.shape[0] * encrypted_image.shape[1]
encrypted_normalized_histograms = [hist / encrypted_total_pixels for hist in encrypted_histograms]

# Calculate the entropy for each channel of the plain image
for plain_normalized_hist in plain_normalized_histograms:
    plain_entropy = -np.sum(plain_normalized_hist * np.log2(plain_normalized_hist + 1e-10))
    plain_entropies.append(plain_entropy)

# Calculate the entropy for each channel of the encrypted image
for encrypted_normalized_hist in encrypted_normalized_histograms:
    encrypted_entropy = -np.sum(encrypted_normalized_hist * np.log2(encrypted_normalized_hist + 1e-10))
    encrypted_entropies.append(encrypted_entropy)

# Compute the global information entropy for each image
plain_global_entropy = np.mean(plain_entropies)
encrypted_global_entropy = np.mean(encrypted_entropies)

# Compare the global information entropies
if encrypted_global_entropy > plain_global_entropy:
    print("The encryption increased the entropy of the image.")
elif encrypted_global_entropy < plain_global_entropy:
    print("The encryption decreased the entropy of the image.")
else:
    print("The encryption did not significantly affect the entropy of the image.")

