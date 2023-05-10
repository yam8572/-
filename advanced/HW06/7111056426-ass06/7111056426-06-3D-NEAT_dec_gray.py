import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名

def gcd(a:int, b:int):
    while b:
        a, b = b, a % b
    return a

def decimal_to_binary(decimal_number:int):
    # Convert decimal to binary
    binary_list = []
    while decimal_number > 0:
        remainder = decimal_number % 2
        binary_list.append(remainder)
        decimal_number //= 2
    
    # Reverse the binary list to get the correct order
    binary_list.reverse()

    # Pad the binary list with leading zeros if necessary
    while len(binary_list) < 8:
        binary_list.insert(0, 0)
    return binary_list

def binary_to_decimal(binary_array):
    # Convert binary array to string
    binary_string = ''.join(str(bit) for bit in binary_array)
    decimal_number = 0
    binary_digits = str(binary_string)
    # reverse the digit ex. 1 1 0 0 >> 0 0 1 1
    binary_digits = binary_digits[::-1]
    for i in range(len(binary_digits)):
        if binary_digits[i] == '1':
            decimal_number += 2**i
    return decimal_number
# binary_to_decimal([0, 1, 1, 0, 0, 1, 0, 0])

if __name__ == '__main__':

    bx, by, bz, rx, ry, rz = input("Enter bx, by, bz, rx, ry, rz: ").split()
    bx = int(bx)
    by = int(by)
    bz = int(bz)
    rx = int(rx)
    ry = int(ry)
    rz = int(rz)

    global src_rgb

    # 印出所有圖片
    DATASRC = 'encrypt/'
    DATATRG = 'decryp/'

    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)

    for i in range (len(src_name)):
            
        # 圖片完整路徑
        src_path = DATASRC + src_name[i]
        print(src_path)
        # Load the image
        src_decimal_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        src_decimal_rgb = cv2.cvtColor(src_decimal_bgr,cv2.COLOR_BGR2RGB)
        print(src_decimal_rgb)
        # N 為影像之水平像素數量，M 為影像之垂直像素數量
        M, N, C = src_decimal_rgb.shape
        
        # Convert the image to grayscale
        src_decimal_gray = cv2.cvtColor(src_decimal_bgr, cv2.COLOR_BGR2GRAY)
        sub1 = src_decimal_bgr[:, :, 0] - src_decimal_gray
        sub2 = src_decimal_bgr[:, :, 1] - src_decimal_gray
        sub3 = src_decimal_bgr[:, :, 2] - src_decimal_gray
        #判斷sub中的每个值是不是都等於0
        if((sub1 == 0).all() and (sub2 == 0).all() and (sub3 == 0).all()):
            K=8
            print("The image is grayscale")
        else:
            K=24
            print("The image is color")

        # src_binary_rgb=np.unpackbits(src_decimal_rgb, axis=-1)
        # Convert the image to binary representation
        src_binary_rgb = np.ndarray((M,N,K), dtype=np.uint8)
        for m in range(M):
                for n in range(N):
                    # Get the BGR values of the pixel
                    r, g, b = src_decimal_rgb[m, n]
                    # Convert each value to binary
                    binary_r = decimal_to_binary(r)
                    binary_g = decimal_to_binary(g)
                    binary_b = decimal_to_binary(b)
                    
                    # Replace the original BGR values with the binary values
                    # 灰階只需一個channel (r,g,b 3 channel 同值)
                    src_binary_rgb[m, n, 0:8] = binary_r
                    if(K==24):
                        src_binary_rgb[m, n, 8:16] = binary_g
                        src_binary_rgb[m, n, 16:24] = binary_b

        # inital variable
        cz = int(rz * (M / gcd(N,M)))
        cx = int(rx * (K / gcd(M,K)))
        cy = int(ry * (N / gcd(K,N)))
        L=5

        decryp_binary_rgb = np.ndarray(src_binary_rgb.shape, dtype=np.uint8)
        # for l in range(L):
        for n in range(N):
            for m in range(M):
                for k in range(K):
                    init_pos = [n,m,k]
                    temp_1 = np.array(init_pos)
                    temp_1[1] = init_pos[1]
                    temp_1[0] = (init_pos[0]-cy*init_pos[2]) % N
                    temp_1[2] = (init_pos[2]-by*temp_1[0]) % K
                    temp_2 = np.array(temp_1)
                    temp_2[0] = temp_1[0]
                    temp_2[2] = (temp_1[2] - cx * temp_1[1]) % K
                    temp_2[1] = (temp_1[1] - bx * temp_2[2]) % M
                    new_pos = np.array(temp_2)
                    new_pos[2] = temp_2[2]
                    new_pos[1] = (temp_2[1] - cz * temp_2[0]) % M
                    new_pos[0] = (temp_2[0] - bz * new_pos[1]) % N
                    decryp_binary_rgb[new_pos[1],new_pos[0],new_pos[2]]=src_binary_rgb[init_pos[1],init_pos[0],init_pos[2]]

        # decryp_decimal_rgb = np.packbits(decryp_binary_rgb, axis=-1)
        # Convert the binary image to decimal representation
        decryp_decimal_rgb = np.ndarray(src_decimal_rgb.shape, dtype=np.uint8)
        for m in range(M):
            for n in range(N):
                # Convert each value to decimal
                dec_r = binary_to_decimal(decryp_binary_rgb[m, n, 0:8])
                if(K==24):
                    dec_g = binary_to_decimal(decryp_binary_rgb[m, n, 8:16])
                    dec_b = binary_to_decimal(decryp_binary_rgb[m, n, 16:24])
                # print(dec_r,dec_g,dec_b)
                # Replace the original BGR values with the binary values
                # 灰階只需一個channel (r,g,b 3 channel 同值)
                if(K==24):decryp_decimal_rgb[m, n]=[dec_r,dec_g,dec_b]
                else:decryp_decimal_rgb[m, n]=[dec_r,dec_r,dec_r]

        # 去掉副檔名
        src_name_temp= Path(src_path).stem
        src_name_temp=src_name_temp.rstrip("enc")
        # 存顏色轉換結果圖
        decryp_decimal_bgr = cv2.cvtColor(decryp_decimal_rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite( DATATRG + src_name_temp + 'dec.png', decryp_decimal_bgr)
