import LaplaceLM.py
#import KneserNeyLM.py

for n in range(2,4):
    lm = LaplaceLM(train_set, n)
    lm.test(test_set)
    predict_sequences = ["Show a great", "I would report", "I would like"]
    for predict_sequence in predict_sequences:
        lm.predict(predict_sequence)
    lm.eval_measures(test_set)
