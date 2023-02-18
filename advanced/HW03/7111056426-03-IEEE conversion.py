import pandas as pd
import os # use directory & join path
from pathlib import Path # 去掉副檔名
import struct

getBin = lambda x: x > 0 and str(bin(x))[2:] or "-" + str(bin(x))[3:]
 
def floatToBinary64(value):
    val = struct.unpack('Q', struct.pack('d', value))[0]
    return getBin(val)
 
def binaryToFloat(value):
    hx = hex(int(value, 2))   
    return struct.unpack("d", struct.pack("q", int(hx, 16)))[0]

def dec_to_bin_csv():
    DATASRC = 'feature/dec/'
    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)
    for i in range (len(src_name)):

        # 檔案完整路徑
        src_path = DATASRC + src_name[i]
        data = pd.read_csv(src_path,header=None)
        
        mean = [floatToBinary64(float(data[1][1])),floatToBinary64(float(data[2][1])),floatToBinary64(float(data[3][1]))]
        std = [floatToBinary64(float(data[1][2])),floatToBinary64(float(data[2][2])),floatToBinary64(float(data[3][2]))]
        MeanStd = [mean,std]
        MeanStd_table = pd.DataFrame(MeanStd)
        col_name = ['red','green','blue']
        row_name = ['Mean','Standard Deviation']
        MeanStd_table.columns = col_name
        MeanStd_table.index = row_name
        # print(MeanStd_table)

        # 匯出 answer table 成 csv 檔
        src_name_temp= Path(src_name[i]).stem 
        src_name_tempp = src_name_temp.rstrip('dec')
        MeanStd_table.to_csv('output/bin/'+ src_name_tempp + 'bin.csv')

def bin_to_dec_csv():
    DATASRC = 'feature/bin/'
    # 抓目錄下所有圖檔檔名
    src_name = os.listdir(DATASRC)
    for i in range (len(src_name)):

        # 檔案完整路徑
        src_path = DATASRC + src_name[i]
        data = pd.read_csv(src_path,header=None)

        mean = [binaryToFloat(data[1][1]),binaryToFloat(data[2][1]),binaryToFloat(data[3][1])]
        std = [binaryToFloat(data[1][2]),binaryToFloat(data[2][2]),binaryToFloat(data[3][2])]
        MeanStd = [mean,std]
        MeanStd_table = pd.DataFrame(MeanStd)
        col_name = ['red','green','blue']
        row_name = ['Mean','Standard Deviation']
        MeanStd_table.columns = col_name
        MeanStd_table.index = row_name
        # print(MeanStd_table)

        # 匯出 answer table 成 csv 檔
        src_name_temp= Path(src_name[i]).stem 
        src_name_tempp = src_name_temp.rstrip('bin')
        MeanStd_table.to_csv('output/dec/'+ src_name_tempp + 'dec.csv')

if __name__ == '__main__':
    choice=0
    while(choice != 3):
        print("選單 1: Decimal to IEEE-754 按 1")
        print("選單 2: IEEE-754 to Decimal 按 2")
        print("選單 3: 結束程式 按 3")
        choice = input("請選擇：")
        choice = int(choice)
        if(choice == 1):
            dec_to_bin_csv()
            print("Decimal to IEEE-754 完成 ! 請至 output 資料夾下 bin 資料夾看轉換結果。")
        elif(choice == 2):
            bin_to_dec_csv()
            print("IEEE-754 to Decimal 完成 ! 請至 output 資料夾下 dec 資料夾看轉換結果。")

        elif(choice == 3):
            print("結束程式")
        elif(choice != 1 and choice != 2 and choice != 3):
            print("無此選單，請依選單輸入 1 2 3")