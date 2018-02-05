import re
from nltk import sent_tokenize, ngrams
from math import log
import random
from LaplaceLM import LaplaceLM

class InterpolationLM:

    def __init__(self, corpus, n=3, threshold=5, lambda1=0.1):
        self.lm1 = LaplaceLM(corpus, n)
        self.lm2 = LaplaceLM(corpus, n-1)
        self.n = n
        self.threshold = threshold
        self.vocabulary = []
        self.corpus = self._preprocess(corpus)
        self.vocabulary = self._train(self.corpus)
        self.lambda1 = lambda1

    def _get_special_tokens(self):
        special_tokens = []
        for i in range(self.n-1):
            special_tokens.append('*START{}*'.format(i+1))
        special_tokens.append('*END*')
        special_tokens.append('*UNK*')
        return special_tokens

    def _clear_sentence(self, sentence):
        return re.sub(r'[.,\/#?!$%\^&\*;:{}=\-_`~()\'\"\n]', '', sentence.lower())

    def _add_tokens(self, sentence):
        start = ''
        for i in range(self.n-1):
            start += '*START{}* '.format(i+1)
        return start + sentence + ' *END*'

    def _preprocess(self, raw_corpus):
        corpus = []
        for line in raw_corpus:
            sentences = sent_tokenize(line)
            for sent in sentences:
                sent = self._clear_sentence(sent)
                sent = self._add_tokens(sent)
                corpus += [sent]
        corpus = ' '.join(corpus)
        distinct_words = set(corpus.split())
        if(self.vocabulary):
            oov_words = [word for word in distinct_words if word not in self.vocabulary]
        else:
            oov_words = [word for word in distinct_words if corpus.count(' ' + word + ' ') < self.threshold]
        for word in oov_words:
            corpus = corpus.replace(' ' + word + ' ', ' *UNK* ')
        corpus = re.sub(' +', ' ', corpus)
        return corpus

    def _train(self, corpus):
        distinct_words = set(corpus.split())
        oov_words = [word for word in distinct_words if corpus.count(' ' + word + ' ') < self.threshold]
        return [word for word in distinct_words if word not in oov_words]

    def _get_probability(self, n_gram):
        return self.lambda1*self.lm1._get_probability(n_gram) + (1-self.lambda1)*self.lm2._get_probability(n_gram[:len(n_gram)-1])

    def _get_random_sentence(self, corpus):
        k = random.randint(0, corpus.count('*START1*')-1)
        pattern = re.compile('\*START1\*.*?\*END\*')
        return re.findall(pattern, corpus)[k]

    def _generate_random_sentence(self, length):
        words = random.sample(self.vocabulary, length)
        sentence = ' '.join(words)
        start = ''
        for i in range(self.n-1):
            start += '*START{}* '.format(i+1)
        sentence = start + sentence + ' *END*'
        return sentence

    def test(self, corpus):
        test_sequences = []
        test_corpus = self._preprocess(corpus)
        correct_sentence = self._get_random_sentence(test_corpus)
        test_sequences += [correct_sentence]
        for i in range(3):
            test_sequences += [self._generate_random_sentence(len(correct_sentence.split())-self.n)]
        for seq in test_sequences:
            print(seq)
            print("\tInterpolation with lambda1: {}:".format(self.lambda1))
            sum = 0
            for n_gram in ngrams(seq.split(' '), self.n):
                prob = self._get_probability(n_gram)
                sum += log(prob)
                print("\t\t", n_gram, prob)
            print("\tProbability is {}\n".format(sum))

    def predict(self, sequence, results=5):
        print(sequence)
        candidates = []
        sequence = self._clear_sentence(sequence)
        sequence = self._add_tokens(sequence)
        print("\t{}-grams:".format(self.n))
        for word in self.vocabulary:
            if word in self._get_special_tokens():
                continue
            seq = re.sub("\*END\*", word, sequence)
            n_gram = (seq.split())[-self.n:]
            prob = self._get_probability(n_gram) * 100
            candidates += [(word, prob)]
        candidates = sorted(candidates, key= lambda tup: tup[1])
        for (word, prob) in candidates[-results:]:
            print("\t\t{}: {:4.2f}%".format(word, prob))

    def eval_measures(self, corpus):
        entropy = 0
        perplexity = 0
        N = 0
        test_sequences = []
        test_corpus = self._preprocess(corpus)
        test_sequences = [e+'*END*' for e in test_corpus.split('*END*') if e]
        for seq in test_sequences:
            seq = re.sub(' \*START1\*', '*START1*', seq)
            for n_gram in ngrams(seq.split(' '), self.n):
                prob = self._get_probability(n_gram)
                entropy += log(prob)
                N += 1
        entropy = -entropy/N
        perplexity = 2**entropy
        print("Entropy is {}".format(entropy))   
        print("Perplexity is {}".format(perplexity))
        return
