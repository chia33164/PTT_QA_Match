import jieba

jieba.set_dictionary('extra_dict/dict.txt.big')
stopword_set = set()
with open('extra_dict/stop_words.txt', 'r', encoding='utf-8') as stopwords:
    for stopword in stopwords:
        stopword_set.add(stopword.strip('\n'))

output = open('ptt_QA_seg.txt', 'w', encoding='utf-8')

with open('Gossiping-QA-Dataset.txt', 'r', encoding='utf-8') as content:
    for texts_num, line in enumerate(content):
        line = line.strip('\n')
        # line = Convert('zh-hant').convert(line)
        # line = line
        words = jieba.cut(line, cut_all=True)
        for word in words:
            if word not in stopword_set:
                output.write(word + ' ')
        output.write('\n')
        if (texts_num + 1) % 10000 == 0:
            print("已完成前 %d 行的斷詞" % (texts_num + 1))
