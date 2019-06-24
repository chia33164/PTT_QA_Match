# PTT_QA_Match

### Install 
```
$ pip install jieba
```

### create_dict.py

來將原本的 ptt 問答做斷詞，並去除掉一些 stopwords
將結果存到 ptt_QA_seg.txt

先設定 jieba 的字典
存下 stopwords 到 stopwordset,以利斷詞時排除不需要的詞

把 Gossiping-QA-Dataset.txt 中得內容逐行取出
並利用 jieba 將該行斷詞斷開
斷開後的詞再經過排除 stopwords 的動作才把斷詞結果存入 ptt_QA_seg.txt 中，作為訓練模型的 dataset
每個斷詞以空格格開

### train_model.py

讀取剛剛利用 create_dict 建出來的 ptt_QA_seg.txt 作為 train model 的 dataset
logging : 印出訓練模型的進度
利用 word2vec 訓練 vector model

參數 :
size 維度
Window 關聯附近的詞的個數
Worker 跑得 thread
Sg 決定使用哪個模型
Min_count 詞出現幾次會把它做成向量
Iter 跑得次數
	
訓練完後儲存成 ptt_QA.word2vec_50.bin

### answer.py

Function avg_feature_vector :

  把送入的詞利用空格斷開
  一個一個去找各個斷詞分別對應到 model 中得哪個 index 並取出其代表的 vector
  他們全部加起來算出來的平均就是該句子的 vector

Function Compute :

  把整個題目用 \t 切開，即可切成 題目和四個選項
  先將題目利用 jieba 斷詞
  斷詞後排除 stopword
  將結果傳入 avg_feature_vector 來得到題目對應的向量
  存成 s1_afv
  用相同的方式取出四個選項對應的向量
  並分別利用 spatial.distance.cosine 和題目做相似度
  並分別把選項的編號和該選項與題目的相似度關聯，放入 res 這個 list 中
  在 4 個選項都放入 res 後
  對 res 裡面的相似度作排序，由大排到小
  藉此只要取出 res 的中第一個 element 的選項編號即是計算出的答案


Function main:

  設定 jieba 的字典
  一樣把 stopword 存下來
  載入剛剛利用 train_model.py 訓練出來的 model
  取得 model 中斷詞的 index
  把 question.txt 中的題目和選項一行一行的丟入 compute 來得到運算的答案
  再把結果寫入檔案中
  
**Commend line**

```
$ python3 create_dict.py
```
產生分詞檔案

```
$ python3 train_model.py
```
訓練模型

```
$ python3 answer.py
```
進行 QA 解答

