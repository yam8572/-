import numpy as np # array operations
import cv2 #opencv library read img operations
import os # use directory & join path
from pathlib import Path # 去掉副檔名

def gcd(a:int, b:int):
    while b:
        a, b = b, a % b
    return a

if __name__ == '__main__':

    bx, by, bz, rx, ry, rz = input("Enter bx, by, bz, rx, ry, rz: ").split()
    bx = int(bx)
    by = int(by)
    bz = int(bz)
    rx = int(rx)
    ry = int(ry)
    rz = int(rz)

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
        src_dec_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)
        src_dec_rgb = cv2.cvtColor(src_dec_bgr,cv2.COLOR_BGR2RGB)
        # N 為影像之水平像素數量，M 為影像之垂直像素數量
        M, N, C = src_dec_rgb.shape
        
        # # Convert the image to grayscale >> 不判斷是否灰階，全用K=24運算
        # src_dec_gray = cv2.cvtColor(src_dec_bgr, cv2.COLOR_BGR2GRAY)
        # sub1 = src_dec_bgr[:, :, 0] - src_dec_gray
        # sub2 = src_dec_bgr[:, :, 1] - src_dec_gray
        # sub3 = src_dec_bgr[:, :, 2] - src_dec_gray
        # #判斷sub中的每个值是不是都等於0
        # if((sub1 == 0).all() and (sub2 == 0).all() and (sub3 == 0).all()):
        #     K=8
        #     print("The image is grayscale")
        # else:
        #     K=24
        #     print("The image is color")

        # covert image pixel decimal to binary
        src_binary_rgb=np.unpackbits(src_dec_rgb, axis=-1)

        K=24
        # inital variable
        cz = int(rz * (M / gcd(N,M)))
        cx = int(rx * (K / gcd(M,K)))
        cy = int(ry * (N / gcd(K,N)))
        L=5

        decryp_binary_rgb = np.ndarray(src_binary_rgb.shape, dtype=np.uint8)
        for l in range(L):
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

        # covert image pixel binary to decimal
        decryp_decimal_rgb = np.packbits(decryp_binary_rgb, axis=-1)

        # 去掉副檔名
        src_name_temp= Path(src_path).stem
        src_name_temp=src_name_temp.rstrip("enc")
        # 存顏色轉換結果圖
        decryp_decimal_bgr = cv2.cvtColor(decryp_decimal_rgb,cv2.COLOR_RGB2BGR)
        cv2.imwrite( DATATRG + src_name_temp + 'dec.png', decryp_decimal_bgr)
