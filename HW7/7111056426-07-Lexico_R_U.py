import numpy as np # array operations
import pandas as pd # 匯出 csv
from itertools import permutations 

# Get all permutations of list_arr
def permutation(list_arr):
    perm = permutations(sorted(list_arr)) 
    perm_list=[]
    rank=0  
    # Print the obtained permutations 
    for i in list(perm):
        temp=""
        for j in range(len(i)):
            if(j!=len(i)-1):temp += str(i[j])+" "
            else: temp+=str(i[j])
        perm_list.append([rank+1,rank,temp])
        rank+=1

    perm_table = pd.DataFrame(perm_list)
    perm_table.columns = ["No","Rank","List"]
    # 匯出 answer table 成 csv 檔
    perm_table.to_csv(f"permTable_{sorted(list_arr)}.csv",index=False)
    perm_table=perm_table.to_string(index=False)
    
    return perm_table

# print(permutation([3,1,5]))

def permutation_rank(p):
    """Given a permutation of {0,..,n-1} find its rank according to lexicographical order

       :param p: list of length n containing all integers from 0 to n-1
       :returns: rank between 0 and n! -1
       :beware: computation with big numbers
       :complexity: `O(n^2)`
    """
    n = len(p)
    fact = 1                                 # compute (n-1) factorial
    for i in range(2, n):
        fact *= i
    # print("fact1",fact)
    r = 0                                    # compute rank of p
    digits = sorted(p)                       # all yet unused digits
    # print("1",digits)
    for i in range(n-1):                     # for all digits except last one
        q = digits.index(p[i])
        # print("q",q)
        r += fact * q
        # print("r",r)

        del digits[q]                        # remove this digit p[i]
        # print("2",digits)

        fact //= (n - 1 - i)                 # weight of next digit //=整數相除同時指派
        # print("fact2",fact)
    return r

def rank_permutation(r, n, list_arr):
    """Given r and n find the permutation of {0,..,n-1} with rank according to lexicographical order equal to r

       :param r n: integers with 0 ≤ r < n!
       :returns: permutation p as a list of n integers
       :beware: computation with big numbers
       :complexity: `O(n^2)`
    """
    fact = 1                                # compute (n-1) factorial
    for i in range(2, n):
        fact *= i
    # digits = list(range(n))               # all yet unused digits
    digits = list(sorted(list_arr))                 # all yet unused digits

    p = []                                  # build permutation
    for i in range(n):
        q = r // fact                       # by decomposing r = q * fact + rest
        r %= fact
        p.append( digits[q] )
        del digits[q]                       # remove digit at position q
        if i != n - 1:
            fact //= (n - 1 - i)            # weight of next digit
    return p
# rank_permutation(2, 3, ['3','2','1'])

if __name__ == '__main__':
    n=0
    pre_n=-1
    type=''
    rank=0
    perm_list=[]
    while True:
        n=input("Input number of elements(N>0):")
        n=int(n)
        if(n<=0):break
        type=''
        rank=0
        perm_list=[]
        while type!='r' and type!='u':
            type=input("input r:Ranking or u:Unranking:")
            if(n!=pre_n):
                # list 預設為[0 ,... ,n]
                list_arr=np.arange(0,n,dtype=np.uint8)
            if(type=='u'):
                pre_n=n
                for i in range(n):
                    list_arr[i]=input(f"input list[{i}]:")
                    list_arr[i]=int(list_arr[i])

                # print permutation table
                perm_table=permutation(list_arr)
                print(perm_table)

                # 輸出 查詢 list 的 rank 結果
                r = permutation_rank(list_arr)

                # user input 參數
                print(f'n={n},type={type},list={list_arr}') 
                print(f"Rank: {r}")

            elif(type=='r'):
                pre_n=n
                rank_num=input("input Rank:")
                rank_num=int(rank_num)

                # print permutation table
                perm_table=permutation(list_arr)
                print(perm_table)

                # 輸出 依 rank 查詢的 list
                p=rank_permutation(rank_num, n, list_arr)
                # user input 參數
                print(f'n={n},type={type},rank_num={rank_num}')
                print(f"Lexicographic Order List: {p}")

