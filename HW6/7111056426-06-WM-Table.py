import numpy as np # array operations
import math
# sort table by R and SE
from operator import itemgetter
import pandas as pd # 匯出 csv

# maxLoop迴圈之最終上限 : -v ~ +v
# n:n 個迴圈 w[w1,w2,...wn]
def IterativeNestedLoop(v,n,w,M):
    table=[]
    ID=1
    slots=np.full(n,-v,dtype='int8')
    
    index = n-1
    while(True):
        # print(slots)
        
        # 計算餘值變動表(Residue Table)
        # R(i, j)= (i, j)∙ (1, 2) mod 5
        R = np.sum(slots*w) % M
        # SE(i, j)= 𝑖 平方 + 𝑗 平方
        SE = np.sum(np.power(slots,2))
        # [r1=i=slots[i],r2=j=slots[j],R,SE]
        table.append([ID,slots.copy(),R,SE])
        ID+=1
        slots[n-1]+=1
        while(slots[index]==v+1):
            if(index==0):
                # stop end for loop
                return table;
            # last index restore -v then next round              
            slots[index:]=-v

            index-=1 
            slots[index]+=1
        index=n-1

def outputPACsv(distortion_table,n,M,w):
    # PA table array
    PA = []
    
    row0 = ['PA',n , M]
    row1 = ['IX', 'd', 'SE']
    row1.extend(w) # ['IX', 'd', 'SE', 'w1',...'wn']

    file_name = f"PA_{n}_{M}_("
    for i in range(n):
        row0.append(f'w{i+1}') # ['PA', 'n', 'M', 'w1',...'wn']
        if(i==n-1):file_name+=f'{w[i]}'
        else:file_name+=f'{w[i]}_'
    file_name+=").csv"
    PA.append(row1)

    temp_rows=[]
    TSE=0
    for IX in range(M):
        TSE+=distortion_table[IX][0][1]
        temp_rows.append([IX,distortion_table[IX][0][0],distortion_table[IX][0][1]])
        for j in range(n):
            temp_rows[IX].append(distortion_table[IX][0][2][j])
    PA.extend(temp_rows)

    # MSE in n pixels:TSE/M/n
    MSE=np.around(TSE/M/n,4)
    PSNR=np.around(10 * np.log10(255**2/MSE),2)

    PA.append(['','TSE',TSE])
    PA.append(['','MSE',MSE])
    PA.append(['','PSNR',PSNR])

    # print(PA)

    PA_table = pd.DataFrame(PA)
    PA_table.columns = row0
    # 匯出 answer table 成 csv 檔
    PA_table.to_csv(file_name,index=False)

def outputHACsv(homogeneous_table,n,M,w):
    # HA table array
    HA = []
    
    row0 = ['HA',n , M]
    row1 = ['IX', 'd', 'SE']
    row1.extend(w) # ['IX', 'd', 'SE', 'w1',...'wn']

    file_name = f"HA_{n}_{M}_("
    for i in range(n):
        row0.append(f'w{i+1}') # ['HA', 'n', 'M', 'w1',...'wn']
        if(i==n-1):file_name+=f'{w[i]}'
        else:file_name+=f'{w[i]}_'
    file_name+=").csv"
    HA.append(row1)

    temp_rows=[]
    temp_weight=[]
    IX=0
    for d in range(M):
        for i in range(len(homogeneous_table[d])):
            temp_rows.append([IX,homogeneous_table[d][i][0],homogeneous_table[d][i][1]])
            IX+=1
            temp_weight.append(homogeneous_table[d][i][2])
    for k in range(len(temp_rows)):
        for j in range(n):
            temp_rows[k].append(temp_weight[k][j])

    HA.extend(temp_rows)
    # print(HA)
    HA_table = pd.DataFrame(HA)
    HA_table.columns = row0
    # 匯出 answer table 成 csv 檔
    HA_table.to_csv(file_name,index=False)

if __name__ == '__main__':
    n=0
    M=0
    
    while n<2 or n>6:
        n = input("n≥2, n≤6 為正整數 n: number of pixels in a cluster(n):")
        n = int(n)
    while M<2 or M>1024:
        M = input("M≥2, M≤1024 為正整數 M: M 代表秘密訊息為 M 進制:")
        M = int(M)
    w=np.zeros(n,dtype=np.uint8)
    for i in range(n):
        w[i] = input(f"w[{i}]:")
        w[i] = int(w[i])

    q = np.round(math.pow(M,1/n) - 1,6)
    v = math.ceil(math.sqrt(math.pow(q,2) * n))
    # print(q,v)

    table=IterativeNestedLoop(v=v,n=n,w=w,M=M)
    # 依照 R 來排序，若相同，則依照 SE 由小而大排序
    table=sorted(table, key = itemgetter(2, 3))   
    # table[index][0]:ID
    # table[index][1]:[r1,r2]
    # table[index][2]:[R] = d
    # table[index][3]:[SE]

    homogeneous_table={} # Homogeneous Alternation Table:HA
    distortion_table={} # Pixel Alternation Table:PA
    for d in range(int(M)):
        homogeneous_table[d]=[]
        distortion_table[d]=[]
    for i in range(len(table)):
        temp_arr=[item for item in table[i][1]]
        if(table[i][2]!=table[i-1][2]):
            distortion_table[table[i][2]].append([table[i][2],table[i][3],temp_arr])
        else:
            homogeneous_table[table[i][2]].append([table[i][2],table[i][3],temp_arr])

    outputPACsv(distortion_table,n,M,w)
    outputHACsv(homogeneous_table,n,M,w)  