import LaplaceLM.py
#import KneserNeyLM.py

def read_corpus_file(path, start, end):  
    file = open(path, 'r', encoding="utf8")
    content = list(file)[start:end]
    file.close()
    return content

filepath = "/home/kostas/Desktop/Semester 2/text engineering analytics/assignments/assignment1/el-en/europarl-v7.el-en.en"
train_set = read_corpus_file(filepath, 0, 5000)
test_set = read_corpus_file(filepath, 5000, 7000)

for n in range(2,4):
    lm = LaplaceLM(train_set, n)
    lm.test(test_set)
    predict_sequences = ["Show a great", "I would report", "I would like"]
    for predict_sequence in predict_sequences:
        lm.predict(predict_sequence)
    lm.eval_measures(test_set)
