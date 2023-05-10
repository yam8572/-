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
    DATASRC = 'source/'
    DATATRG = 'encrypt/'

    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)
    # Open a file for writing
    with open('parame/parameter.txt', 'w') as f:

        for i in range (len(src_name)):
                
            # 圖片完整路徑
            src_path = DATASRC + src_name[i]
            print(src_path)
            # 去掉副檔名
            src_name_temp= Path(src_path).stem

            # Load the image
            src_decimal_bgr = cv2.imread(src_path, cv2.IMREAD_COLOR)

            # # Convert the image to grayscale >> 不判斷是否灰階，全用K=24運算
            # src_dec_gray = cv2.cvtColor(src_decimal_bgr, cv2.COLOR_BGR2GRAY)
            # sub1 = src_decimal_bgr[:, :, 0] - src_dec_gray
            # sub2 = src_decimal_bgr[:, :, 1] - src_dec_gray
            # sub3 = src_decimal_bgr[:, :, 2] - src_dec_gray
            # #判斷sub中的每个值是不是都等於0
            # if((sub1 == 0).all() and (sub2 == 0).all() and (sub3 == 0).all()):
            #     K=8
            #     print("The image is grayscale")
            # else:
            #     K=24
            #     print("The image is color")
            
            # 原為 BGR 轉為 RGB
            src_decimal_rgb = cv2.cvtColor(src_decimal_bgr,cv2.COLOR_BGR2RGB)
            
            # N 為影像之水平像素數量，M 為影像之垂直像素數量
            M, N, C = src_decimal_rgb.shape
            K=24
            
            # # inital variable
            cz = int(rz * (M / gcd(N,M)))
            cx = int(rx * (K / gcd(M,K)))
            cy = int(ry * (N / gcd(K,N)))

            # inital transform matrix
            mattix_1 = np.array([[1, bz, 0], [cz, 1+bz*cz, 0], [0, 0, 1]])
            mattix_2 = np.array([[1, 0, 0], [0, 1, bx], [0, cx, 1+bx*cx]])
            mattix_3 = np.array([[1+by*cy, 0, cy], [0, 1, 0], [by, 0, 1]])

            # N, M, K, gcd(N, M), gcd(M, K), gcd(K, N), 𝑏𝑥, 𝑏𝑦,  𝑏𝑧,  𝑟𝑥,  𝑟𝑦, 𝑟𝑧, 𝑐𝑧,  𝑐𝑥, 𝑐y
            f.write('N='+str(N)+'\n')
            f.write('M='+str(M)+'\n')
            f.write('K='+str(K)+'\n')
            f.write('gcd(N, M)='+str(gcd(N, M))+'\n')
            f.write('gcd(M, K)='+str(gcd(M, K))+'\n')
            f.write('gcd(K, N)='+str(gcd(K, N))+'\n')
            f.write('bx='+str(bx)+'\n')
            f.write('by='+str(by)+'\n')
            f.write('bz='+str(bz)+'\n')
            f.write('rx='+str(rx)+'\n')
            f.write('ry='+str(ry)+'\n')
            f.write('rz='+str(rz)+'\n')
            f.write('cz='+str(cz)+'\n')
            f.write('cx='+str(cx)+'\n')
            f.write('cy='+str(cy)+'\n')
            f.write('Sz[0]='+str(mattix_1[0][0])+' '+str(mattix_1[0][1])+' '+str(mattix_1[0][2])+'\n')
            f.write('Sz[1]='+str(mattix_1[1][0])+' '+str(mattix_1[1][1])+' '+str(mattix_1[1][2])+'\n')
            f.write('Sz[2]='+str(mattix_1[2][0])+' '+str(mattix_1[2][1])+' '+str(mattix_1[2][2])+'\n')
            f.write('Sx[0]='+str(mattix_2[0][0])+' '+str(mattix_2[0][1])+' '+str(mattix_2[0][2])+'\n')
            f.write('Sx[1]='+str(mattix_2[1][0])+' '+str(mattix_2[1][1])+' '+str(mattix_2[1][2])+'\n')
            f.write('Sx[2]='+str(mattix_2[2][0])+' '+str(mattix_2[2][1])+' '+str(mattix_2[2][2])+'\n')
            f.write('Sy[0]='+str(mattix_3[0][0])+' '+str(mattix_3[0][1])+' '+str(mattix_3[0][2])+'\n')
            f.write('Sy[1]='+str(mattix_3[1][0])+' '+str(mattix_3[1][1])+' '+str(mattix_3[1][2])+'\n')
            f.write('Sy[2]='+str(mattix_3[2][0])+' '+str(mattix_3[2][1])+' '+str(mattix_3[2][2])+'\n')

            # L (5 ≤ L ≤ 1000)，代表加密的次數
            L=5
            # covert image pixel decimal to binary
            src_binary_rgb=np.unpackbits(src_decimal_rgb, axis=-1)

            enc_binary_rgb = np.ndarray(src_binary_rgb.shape, dtype=np.uint8)
            for l in range(L):
                for m in range(M):
                    for n in range(N):
                        for k in range(K):
                            init_pos=[n,m,k]
                            temp_1=mattix_1.dot(init_pos)
                            temp_1=[temp_1[0]%N,temp_1[1]%M,temp_1[2]%K]
                            temp_2=mattix_2.dot(temp_1)
                            temp_2=[temp_2[0]%N,temp_2[1]%M,temp_2[2]%K]
                            new_pos=mattix_3.dot(temp_2)
                            new_pos=[new_pos[0]%N,new_pos[1]%M,new_pos[2]%K]
                            enc_binary_rgb[new_pos[1],new_pos[0],new_pos[2]]=src_binary_rgb[init_pos[1],init_pos[0],init_pos[2]]
            
            # covert image pixel binary to decimal
            encryp_decimal_rgb = np.packbits(enc_binary_rgb, axis=-1)
            # 存顏色轉換結果圖
            encryp_deciaml_bgr = cv2.cvtColor(encryp_decimal_rgb,cv2.COLOR_RGB2BGR)
            if(i<9):cv2.imwrite( DATATRG + '0'+ str(i+1)+ '_' + src_name_temp + '_enc.png', encryp_deciaml_bgr)
            else:cv2.imwrite( DATATRG + str(i+1)+ '_' + src_name_temp + '_enc.png', encryp_deciaml_bgr)