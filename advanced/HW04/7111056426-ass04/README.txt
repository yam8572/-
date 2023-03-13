暴力法距離皆 使用 Correlation method
二元法 可依選單選距離
依照選單提示輸入距離量測標準
1.Correlation 按 1
2.Chi-Squared 按 2
3.Intersection 按 3
4.Hellinger 按 4

cmd 下指令
暴力法 python 7111056426-04-WCT-Bru.py 執行檔案
二元法 python 7111056426-04-WCT-Bin.py 執行檔案

暴力法結果輸出範例:
每圖停下權重
Distance method is Correlation
1  stop weight: 0.8_0.51_0.49
2  stop weight: 0.0_0.51_0.49
3  stop weight: 0.3_0.45_0.48
4  stop weight: 0.36_0.05_0.68
5  stop weight: 1.0_0.64_0.48
6  stop weight: 0.84_0.55_0.47

每圖最後與 source target 的距離和差值
result_bru/WCT01-Bru-0.8_0.51_0.49.png
src_d= 0.29734103364266257 tar_d= 0.2971779262625108
difference= 0.0001631073801517391

result_bru/WCT02-Bru-0.0_0.51_0.49.png
src_d= 0.7670437036691965 tar_d= 0.7660563920531841
difference= 0.0009873116160123985

result_bru/WCT03-Bru-0.3_0.45_0.48.png
src_d= 0.4437860737674082 tar_d= 0.4477898268716156
difference= 0.004003753104207419

result_bru/WCT04-Bru-0.36_0.05_0.68.png
src_d= 0.7453260450534768 tar_d= 0.7457665302894089
difference= 0.00044048523593209765

result_bru/WCT05-Bru-1.0_0.64_0.48.png
src_d= 0.2105460874260876 tar_d= 0.2107350526810948
difference= 0.0001889652550071952

二元法結果輸出範例: (使用 Correlation method)
每圖停下權重
1  stop weight: 0.5_0.5_0.171875
2  stop weight: 0.16796875_0.5_0.5
3  stop weight: 0.30078125_0.5_0.5
4  stop weight: 0.3125_0.1669921875_0.5
5  stop weight: 0.5_0.5_0.5
6  stop weight: 0.5_0.5_0.25
每圖最後與 source target 的距離和差值
result_bin/WCT01-Bin-0.5_0.5_0.171875.png
src_d= 0.6488365415807459 tar_d= 0.4471450431156695
difference= 0.2016914984650764

result_bin/WCT02-Bin-0.16796875_0.5_0.5.png
src_d= 0.8720365470256464 tar_d= 0.8406706535200543
difference= 0.031365893505592046

result_bin/WCT03-Bin-0.30078125_0.5_0.5.png
src_d= 0.44489018871493785 tar_d= 0.4505844619390841
difference= 0.005694273224146229

result_bin/WCT04-Bin-0.3125_0.1669921875_0.5.png
src_d= 0.8410137104011179 tar_d= 0.7942209792493088
difference= 0.04679273115180915

result_bin/WCT05-Bin-0.5_0.5_0.5.png
src_d= 0.3842016117217702 tar_d= 0.11162795826637216
difference= 0.272573653455398

result_bin/WCT06-Bin-0.5_0.5_0.25.png
src_d= 0.2719629312017611 tar_d= 0.23828700600833985
difference= 0.033675925193421274

暴力法 至 result_bru 看最佳權重色彩轉換圖
二元法 至 result_bin 看最佳權重色彩轉換圖