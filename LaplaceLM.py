import re
from nltk import sent_tokenize, ngrams
from operator import itemgetter
from math import log


class LaplaceLM:

    def __init__(self, corpus_file, n=1, lim=4000, unknown_threshold=10):
        self.n = n
        self.corpus = self._preprocess(self._read_corpus_file(corpus_file, lim))
        self.vocabulary = self._train(unknown_threshold)

    def _read_corpus_file(self, path, lim):  
        file = open(path, 'r', encoding="utf8")
        content = list(file)[:lim]
        file.close()
        return content

    def _preprocess(self, raw_corpus):
        corpus = []
        for line in raw_corpus:
            sentences = sent_tokenize(line)
            for sent in sentences:
                sent = re.sub(r'[.,\/#?!$%\^&\*;:{}=\-_`~()\'\"\n]', '', sent.lower())
                start = ''
                for i in range(self.n-1):
                    start += '*START{}* '.format(i+1)
                sent = start + sent + ' *END*'
                corpus += [sent]
        corpus = ' '.join(corpus)
        return corpus        

    def _train(self, threshold):
        distinct_words = set(self.corpus.split())
        vocabulary = [word if self.corpus.count(word) > threshold else '*UNK*' for word in distinct_words]
        return set(vocabulary)

    def _get_probability(self, n_gram):
        c1 = self.corpus.count(' '.join(n_gram))
        c2 = self.corpus.count(' '.join(n_gram[:len(n_gram)-1]))
        return (c1 + 1) / (c2 + len(self.vocabulary))

    def test(self, sequence):
        print(sequence)
        sequence = self._preprocess([sequence])
        print("\t{}-grams:".format(self.n))
        sum = 0
        for n_gram in ngrams(sequence.split(' '), self.n):
            prob = self._get_probability(n_gram)
            sum += log(prob)
            print("\t\t", n_gram, prob)
        print("\tProbability is {}\n".format(sum)) 

    def entropy(self, sequences):
        entropy = 0
        N = 0
        for seq in sequences:
            seq = self._preprocess([seq])
            for n_gram in ngrams(seq.split(' '), self.n):
                prob = self._get_probability(n_gram)
                entropy += log(prob)
                N += 1
        entropy = -entropy/N
        print("Entropy is {}\n".format(entropy))    

    def predict(self, sequence, results=5):
        print(sequence)
        candidates = []
        sequence = self._preprocess([sequence])
        print("\t{}-grams:".format(self.n))
        for word in self.vocabulary:
            seq = re.sub("\*END\*", word, sequence)
            seq = (seq.split())[-self.n:]
            prob = self._get_probability(seq) * 100
            candidates += [(' '.join(seq), prob)]
        candidates = sorted(candidates, key= lambda tup: tup[1])
        for (word, prob) in candidates[-results:]:
            print("\t\t{}: {:4.2f}%".format(word, prob))



for n in range(2,5):
    lm = LaplaceLM("/home/kostas/Desktop/Semester 2/text engineering analytics/assignments/el-en/europarl-v7.el-en.en",n)
    test_sequences = ["He please god football.", "He plays god football.", "He plays good football.", "He players good football.", "He pleases god ball.", "Of the rules.", "Rules the of."]
    predict_sequences = ["Of the", "I would like to"]

#     for test_sequence in test_sequences:
#         lm.test(test_sequence)

#     lm.entropy(test_sequences)

    for predict_sequence in predict_sequences:
        lm.predict(predict_sequence)
