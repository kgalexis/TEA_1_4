import re
import random
from nltk import ngrams
from math import log2

class KNLM:

    def __init__(self, corpus, n, threshold=5, D=0.1):
        self.D = D
        self.n = n
        self.threshold = threshold
        self.vocabulary = self._train(corpus)

    def _strip(self, text):
        text = re.sub(r'[.,\/#?!$%\^&\*;:{}\[\]=\-_`~()\'\"\n]', ' ', text.lower())
        return re.sub(' +', ' ', text)

    def _add_tokens(self, sequence):
        start = ''
        for i in range(self.n-1):
            start += '*START{}* '.format(i+1)
        new_sequence = start + sequence + ' *END*'
        return re.sub(' +', ' ', new_sequence)

    def _add_unk(self, text, mode):
        distinct_words = set(text.split())
        if mode==0:
            oov_words = [word for word in distinct_words if text.count(' ' + word + ' ') < self.threshold]
        else:
            oov_words = [word for word in distinct_words if word not in self.vocabulary]
        for word in oov_words:
            text = text.replace(' ' + word + ' ', ' *UNK* ')
        text = re.sub(' +', ' ', text)
        return text

    def _train(self, corpus):
        self.train_corpus = [self._strip(sentence) for sentence in corpus]
        self.train_corpus = [self._add_tokens(sentence) for sentence in self.train_corpus]
        self.train_corpus = ' '.join(self.train_corpus)
        self.train_corpus = self._add_unk(self.train_corpus, 0)
        return set(self.train_corpus.split())

    def _get_probability(self, n_gram):
        c1 = self.train_corpus.count(' '.join(n_gram))
        if c1!=0:
            c2 = self.train_corpus.count(' '.join(n_gram[:len(n_gram)-1]))
            return c1 / c2 * self.D
        else:
            reserved_prob = 0
            total_zero_tokens = 0
            zero_tokens = 0
            c_prev = self.train_corpus.count(' '.join(n_gram[:(len(n_gram)-1)])) #count of the sequence without the last word
            c_last = self.train_corpus.count(n_gram[len(n_gram)-1]) #count of the last word
            for word in self.vocabulary:
                nseq = tuple(list(n_gram[:len(n_gram)-1]) + [word])
                c_new = self.train_corpus.count(' '.join(nseq)) #count of the new sequence with new last word
                c_word = self.train_corpus.count(word) #count of new word
                if c_prev!=0:
                    reserved_prob+= c_new/c_prev * self.D #steal and increase reserved_probability by the defined percentage (if new sequence is still not in training, then nothing changes)
                if c_new==0: #if new sequence not in training
                    total_zero_tokens+= c_word # increase total tokens of this word's occurences
                    zero_tokens+= 1 #increase number of tokens that had zero occurences
            prob = reserved_prob / zero_tokens * c_last / total_zero_tokens
            return prob
        #return (c1 + 1) / (c2 + len(self.vocabulary))

    def _get_special_tokens(self):
        special_tokens = ['*START{}*'.format(i+1) for i in range(self.n-1)]
        special_tokens.append('*END*')
        special_tokens.append('*UNK*')
        return special_tokens

    def get_random_sentence(self, corpus):
        return random.choice(corpus)

    def generate_test_sequences(self, length, n_sequences=3):
        sequences = []
        for i in range(n_sequences):
            words = random.sample(self.vocabulary - set(self._get_special_tokens()), length)
            sequence = ' '.join(words)
            sequences.append(sequence)
        return sequences

    def test(self, sequence):
        sequence = self._strip(sequence)
        sequence = self._add_tokens(sequence)
        sequence = self._add_unk(sequence, 1)
        prob_sum = 0
        for n_gram in ngrams(sequence.split(' '), self.n):
            prob = self._get_probability(n_gram)
            prob_sum += log2(prob)
        return prob_sum

    def predict(self, sequence, results=5):
        candidates = []
        sequence = self._strip(sequence)
        sequence = self._add_tokens(sequence)
        for word in self.vocabulary:
            if word in self._get_special_tokens():
                continue
            seq = re.sub("\*END\*", word, sequence)
            n_gram = (seq.split())[-self.n:]
            prob = self._get_probability(n_gram) * 100
            candidates += [(word, prob)]
        candidates = sorted(candidates, key= lambda tup: tup[1])
        for (word, prob) in candidates[-results:]:
            print("\t{}: {:4.2f}%".format(word, prob))
        return

    def evaluate(self, corpus):
        entropy = 0
        N = 0
        for sentence in corpus:
            entropy += self.test(sentence)
            N += len(sentence.split())
        entropy = -entropy/N
        perplexity = 2**entropy
        print("Entropy is {}".format(entropy))   
        print("Perplexity is {}".format(perplexity))
        return
