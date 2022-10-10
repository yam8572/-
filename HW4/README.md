# 計算機圖學與應用 HW04
求解最佳基底 <br>
授課教師：王宗銘<br>
2022/10/05<br>
## 1. 請以 python 程式語言撰寫求解 multiple base (MB) data embedding algorithm 之最佳基底向量(optimal base vector, OBV)。
程式名稱：學號-04-determine OBV.py。<br>
輸入: n, F 均為正整數。若輸入 n=0 或 F=0 則結束程式，否則持續等待下一次輸入 n, F.<br>
n: number of pixels in a cluster<br>
F: target notation<br>
輸出：<br>
1. Optimal Base Vector OBV:
2. Derived Notation M:
3. Difference D:
4. EMSE OBV:
5. PSNR:
![](https://i.imgur.com/KO0usqe.png)

**輸入範例 1:**<br>
Input number of pixels in a cluster (n) and the target notation (F): 3 49<br>
**輸出範例 1**<br>
1. Optimal Base Vector OBV: 3, 4, 5
2. Derived Notation M: 60
3. Difference: 11
4. EMSE OBV: 1.3889
5. PSNR: 46.70

**輸入範例 2:**<br>
Input number of pixels in a cluster (n) and the target notation (F): 5 499409<br>
**輸出範例 2**<br>
1. Optimal Base Vector OBV: 13, 13, 14, 14, 15
2. Derived Notation M: 532350
3. Difference: 32941
4. EMSE OBV: 16.3667
5. PSNR: 35.99

**輸入範例 3:**<br>
Input number of pixels in a cluster (n) and the target notation (F): 6 2589478<br>
**輸出範例 3**<br>
1. Optimal Base Vector OBV: 11, 11, 11, 12, 13, 13
2. Derived Notation M: 2699268
3. Difference: 109790
4. EMSE OBV: 11.6944
5. PSNR: 37.45

**輸入範例 4:**<br>
Input number of pixels in a cluster (n) and the target notation (F): 4 626<br>
**輸出範例 4**<br>
1. Optimal Base Vector OBV: 5, 5, 5, 6
2. Derived Notation M: 750
3. Difference: 124
4. EMSE OBV: 2.2917
5. PSNR: 44.53

## 2. 繳交檔案
(1) python 程式，程式名稱：學號-04-determine optimal base.py。<br>

**講義密碼:**2022CGMSGMB!
