from LaplaceLM import LaplaceLM
from InterpolatedLM import InterpolatedLM
import numpy as np

def read_file(path, start, end):
    with open(path, 'r') as file:
        return list(file)[start:end]

filepath = '/home/kostas/Desktop/Semester 2/text engineering analytics/assignments/assignment1/el-en/europarl-v7.el-en.en'
train_set = read_file(filepath, 0, 3000)
test_set = read_file(filepath, 3000, 4000)

print('\n========== Bigram Language Model ==========')
bigram_lm = LaplaceLM(train_set, 2, 5)

correct_sentence = bigram_lm.get_random_sentence(test_set)
print("{}\n---Log-probability: {}\n".format(correct_sentence, bigram_lm.test(correct_sentence)))
test_sequences = bigram_lm.generate_test_sequences(len(correct_sentence.split()))
for test_sequence in test_sequences:
    print("{}\n\n---Log-probability: {}\n".format(test_sequence, bigram_lm.test(test_sequence)))

predict_sequences = ["Show a great", "This will take", "I would like"]
for predict_sequence in predict_sequences:
    print(predict_sequence)
    bigram_lm.predict(predict_sequence)

bigram_lm.evaluate(test_set)

print('\n========== Trigram Language Model ==========')
trigram_lm = LaplaceLM(train_set, 3, 5)

print("{}\n---Log-probability: {}\n".format(correct_sentence, trigram_lm.test(correct_sentence)))
for test_sequence in test_sequences:
    print("{}\n\n---Log-probability: {}\n".format(test_sequence, trigram_lm.test(test_sequence)))

for predict_sequence in predict_sequences:
    print(predict_sequence)
    trigram_lm.predict(predict_sequence)

trigram_lm.evaluate(test_set)

print('\n========== Interpolated Language Model ==========')
for l in np.linspace(0, 1, 6):
    print('\nl1: {}, l2: {}'.format(l, 1-l))
    interpolated_lm = InterpolatedLM(train_set, l, 5)
    interpolated_lm.evaluate(test_set)
