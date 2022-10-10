#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys, os
import cv2
import numpy as np
import math


# In[2]:


def cal_M(BV):
    M = 1
    for i in range(len(BV)):
        M*=BV[i]
    return M

def EMSE(OBV):
    emse_b = 0.0
    for b in range(len(OBV)):
        emse_b += ((OBV[b]**2)-((-2)**((OBV[b]+1)%2)))/12
    emse_obv = emse_b / len(OBV)
    return round(emse_obv,4)

def main():
    print("Input number of pixels in a cluster (n>=2) and the target notation (F>=4):")
    n, F = map(int, input().split())
    if n==0 or F==0:
        print("exit")
        exit()
    else:
        while(n<2 or F<4):
            if(n<2):
                print("pixels in a cluster (n) should >=2,please retry")
                print("Input number of pixels in a cluster (n>=2) and the target notation (F>=4):")
                n, F = map(int, input().split())
            elif(F<4):
                print("target notation (F>=4),please retry")
                print("Input number of pixels in a cluster (n>=2) and the target notation (F>=4):")
                n, F = map(int, input().split())


        b=math.ceil(F**(1/n))
        FBV=np.full(n,b)
        FBV_M=cal_M(FBV)

        cur_CBV=np.copy(FBV)
        cur_M=FBV_M
        i_front=0
        i_rear=len(cur_CBV)-1
        stop=False
        # record all M stop when M repeat
        M_array=[]
        # record M which M>=F 
        CM_array=[]
        # record EMSE which M>=F 
        CEMSE_array=[]
        # record CBV which M>=F
        CBV_array=[]

        while(stop==False):

            cur_M=cal_M(cur_CBV)
            if(np.any(M_array==cur_M)):
                # repeat
                #print("repeat stop")
                stop=True
            else:
                if(cur_M>=F):
                    CBV_array.append(np.array(cur_CBV))
                    CEMSE_array.append(EMSE(cur_CBV))
                    CM_array.append(cur_M)
                M_array.append(cur_M)

                if(cur_M>F):
                    cur_CBV[i_front]-=1 
                    i_front+=1
                elif(cur_M<F):
                    cur_CBV[i_rear]+=1
                    i_rear-=1
                else:#temp_M==F
                    #print("M=F stop")
                    stop=True

        
        more_CBV = np.copy(FBV)
        more_CBV[0]-=2
        more_M=cal_M(more_CBV)
        if(more_M>=F):
            CBV_array.append(np.array(more_CBV))
            CEMSE_array.append(EMSE(more_CBV))
            CM_array.append(more_M)
        
        best_emse=np.min(CEMSE_array)
        best_index=np.argmin(CEMSE_array)
        best_M=CM_array[best_index]
        best_BV=CBV_array[best_index]
        psnr=round(10 * np.log10(255 * 255 / best_emse),2)
        diff=best_M-F

        print("1.Optimal Base Vector OBV: ",best_BV)
        print("2.Derived Notation M:",best_M)
        print("3.Difference:",diff)
        print("4.EMSE OBV:",best_emse)
        print("5.PSNR",psnr)

        main()


# In[3]:


if __name__ == '__main__':
    main()

