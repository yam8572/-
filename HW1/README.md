# 計算機圖學HW1

計算機圖學與應用 6650<br>
影像分析練習<br>
授課教師：王宗銘<br>
2022/09/08<br>

### 1. 請以 python 程式語言撰寫影像分析程式，可以使用 opencv 套件。
程式名稱：學號-01-image analysis.py。<br>
輸入：在 source 目錄下的彩色影像若干張。<br>
輸出：<br>
(1) 在 result 目錄下的分析結果，兩個檔案 影像名稱-mean-std.csv 與
影像名稱-his.csv。<br>
(2) 在程式執行時，顯示處理影像三個色彩頻道的值方圖。<br>
### 2. 需要分析的項目：
#### 1. 寫入第 1 個檔案，檔案名稱： 影像名稱-mean-std.csv，例如 kodim07-meanstd.csv。
第 1 列為紅色、綠色、藍色頻道之平均值，取小數 2 位，第 3 位四捨五入。<br>
第 2 列為紅色、綠色、藍色頻道之標準差(standard deviation)，取小數 2 位，第
3 位四捨五入。<br>
例如：<br>



|          | R Channel | G Channel | B Channel |
| -------- | --------  | --------  | --------  |
| 平均      | 87.56    | 152.30     | 182.36    |
| 標準差    | 15.23     | 18.45     | 25.36     |

#### 2. 寫入第 2 個檔案，檔案名稱： 影像名稱-his.csv，例如 kodim07-his.csv
第 1 行為紅色頻道 0-255 之個數，共 256 列。<br>
第 2 行為綠色頻道 0-255 之個數，共 256 列。<br>
第 3 行為藍色頻道 0-255 之個數，共 256 列。<br>
例如：

|    | R Channel | G Channel | B Channel |
| ---| --------  | --------  | --------  |
| 0  | 3344      | 7638      | 62867     |
| 1  | 711       | 890       | 1892      |
| 2  | 0         | 0         | 1807      |
...
| 253 | 281     | 56         | 0         |
| 254 | 0       | 45         | 1          |
| 255 | 5379    | 642       | 37          |

#### 3. 提供測試檔案，4 個。
(1) baboon.png<br>
![](https://i.imgur.com/Jcq6Dhd.png)

(2) kodim17.png<br>
![](https://i.imgur.com/DncmvLS.png)

(3) kodim17.png<br>
![](https://i.imgur.com/wM45Ziu.png)

(4) peppers.png<br>
![](https://i.imgur.com/bYKP9Gn.png)
