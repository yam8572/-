Lexicographical Oder Practice
執行 python 7111056426-07-Lexico_R_U.py

Ranking： 給定 list，求出對應的 rank

輸入範例 1:
Input number of elements: 3
Ranking or Unranking: u
input list[0]: 2
input list[1]: 0
input list[2]: 1

輸出範例 1:
 No  Rank  List
  1     0 0 1 2
  2     1 0 2 1
  3     2 1 0 2
  4     3 1 2 0
  5     4 2 0 1
  6     5 2 1 0
Rank: 4

輸入範例 2:
Input number of elements: 3
Ranking or Unranking: u
input list[0]: 3
input list[1]: 1
input list[2]: 5

輸出範例 2:
No  Rank  List
  1     0 1 3 5
  2     1 1 5 3
  3     2 3 1 5
  4     3 3 5 1
  5     4 5 1 3
  6     5 5 3 1
Rank: 2

Unranking： 給定 rank，求出對應的 list

**會依上次 Ranking 所查詢的Rank表 做查詢，
若無用 Ranking 查詢過或與上次 Ranking 查詢的 element長度不同，
預設為 [0,...,n] ex. element=3 list預設為[0,1,2] 所產生的 permutation rank table
**

輸入範例 1:
Input number of elements: 3
Ranking or Unranking: r
Input Rank: 3

輸出範例 1:
 No  Rank  List
  1     0 0 1 2
  2     1 0 2 1
  3     2 1 0 2
  4     3 1 2 0
  5     4 2 0 1
  6     5 2 1 0
Lexicographic Order List: [1, 2, 0]

輸入範例 2:
Input number of elements: 3
Ranking or Unranking: r
Input Rank: 3

輸出範例 2:
No  Rank  List
  1     0 1 3 5
  2     1 1 5 3
  3     2 3 1 5
  4     3 3 5 1
  5     4 5 1 3
  6     5 5 3 1
Lexicographic Order List: [3, 5, 1]

** permutation rank table 有匯出csv檔可供對照 **
