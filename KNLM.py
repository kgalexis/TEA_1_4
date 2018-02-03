import re
from nltk import sent_tokenize, ngrams
from math import log
import random

class KNLM:

    def __init__(self, corpus_file, n=1, start=0, end=4000, unknown_threshold=1, D=0.1):
        self.n = n
        self.D = D
        self.corpus = self._preprocess(self._read_corpus_file(corpus_file, start, end))
        self.train_vocabulary = self._train(unknown_threshold)
        self.corpus = self._UNK(self.corpus)
        self.test_dataset = self._preprocess(self._read_corpus_file(corpus_file, 4000, 6000))
        self.test_sentences = self._test_sentences_splitting(self._read_corpus_file(corpus_file, 4000, 6000))

    def _read_corpus_file(self, path, start, end):  
        file = open(path, 'r', encoding="utf8")
        content = list(file)[start:end]
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

    def _test_sentences_splitting(self, raw_corpus):
        corpus = []
        for line in raw_corpus:
            sentences = sent_tokenize(line)
            for sent in sentences:
                sent = re.sub(r'[.,\/#?!$%\^&\*;:{}=\-_`~()\'\"\n]', '', sent.lower())
                corpus += [sent]
        return corpus       

    def _train(self,threshold):
        distinct_words = set(self.corpus.split())
        vocabulary = [word if self.corpus.count(word) > threshold else '*UNK*' for word in distinct_words]
        return set(vocabulary)

    def _UNK(self, seq):
        seq = [word if word in self.train_vocabulary else '*UNK*' for word in seq.split()]
        return ' '.join(seq)
    
    def _get_probability(self, n_gram):
        c1 = self.corpus.count(' '.join(n_gram))
        if c1!=0:
            c2 = self.corpus.count(' '.join(n_gram[:len(n_gram)-1]))
            return c1 / c2 * self.D
        else:
            reserved_prob = 0
            total_zero_tokens = 0
            zero_tokens = 0
            c_prev = self.corpus.count(' '.join(n_gram[:(len(n_gram)-1)])) #count of the sequence without the last word
            c_last = self.corpus.count(n_gram[len(n_gram)-1]) #count of the last word
            for word in self.train_vocabulary:
                nseq = tuple(list(n_gram[:len(n_gram)-1]) + [word])
                c_new = self.corpus.count(' '.join(nseq)) #count of the new sequence with new last word
                c_word = self.corpus.count(word) #count of new word
                if c_prev!=0:
                    reserved_prob+= c_new/c_prev * self.D #steal and increase reserved_probability by the defined percentage (if new sequence is still not in training, then nothing changes)
                if c_new==0: #if new sequence not in training
                    total_zero_tokens+= c_word # increase total tokens of this word's occurences
                    zero_tokens+= 1 #increase number of tokens that had zero occurences
            prob = reserved_prob / zero_tokens * c_last / total_zero_tokens
            return prob
    
    def get_test_sentence(self):
        return self.test_sentences[random.randint(0, len(self.test_sentences))]
    
    def create_random_sentence(self, length):
        clean_vocabulary = self.train_vocabulary
        vocabulary_words = list(self.train_vocabulary)
        r = re.compile("[*]")
        unwanted_words = list(filter(r.match, vocabulary_words)) 
        for i in unwanted_words:
            clean_vocabulary.remove(i)
        words = random.sample(self.train_vocabulary,length)
        sentence = ' '.join(words)
        return sentence 
        
    def test(self, sequence):
        print(sequence)
        sequence = self._preprocess([sequence])
        sequence = self._UNK(sequence)
        print("\t{}-grams:".format(self.n))
        sum = 0
        for n_gram in ngrams(sequence.split(' '), self.n):
            prob = self._get_probability(n_gram)
            sum += log(prob)
            print("\t\t", n_gram, prob)
        print("\tProbability is {}\n".format(sum)) 

    def eval_measures(self, sequences):
        entropy = 0
        perplexity = 0
        N = 0
        for seq in sequences:
            seq = self._preprocess([seq])
            seq = self._UNK(seq)
            for n_gram in ngrams(seq.split(' '), self.n):
                prob = self._get_probability(n_gram)
                entropy += log(prob)
                N += 1
        entropy = -entropy/N
        perplexity = 2**entropy
        print("Entropy is {}\n".format(entropy))   
        print("Perplexity is {}\n".format(perplexity))  

    def predict(self, sequence, results=5):
        print(sequence)
        candidates = []
        sequence = self._preprocess([sequence])
        sequence = self._UNK(sequence)
        print("\t{}-grams:".format(self.n))
        for word in self.train_vocabulary:
            seq = re.sub("\*END\*", word, sequence)
            seq = (seq.split())[-self.n:]
            prob = self._get_probability(seq) * 100
            candidates += [(' '.join(seq), prob)]
        candidates = sorted(candidates, key= lambda tup: tup[1])
        for (word, prob) in candidates[-results:]:
            print("\t\t{}: {:4.2f}%".format(word, prob))



for n in range(2,3):
    lm = KNLM("C:/Users/zeus8/Desktop/europarl-v7.el-en.en",n)

    test_sequences = []
    test_set_sentence = lm.get_test_sentence()
    test_sequences.append(test_set_sentence)
    for i in range(0,4):
        test_sequences.append(lm.create_random_sentence(len(test_set_sentence.split())))

    predict_sequences = ["Of the", "I would like to"]

    for test_sequence in test_sequences:
        lm.test(test_sequence)

    lm.eval_measures(test_sequences)
    
    for predict_sequence in predict_sequences:
        lm.predict(predict_sequence)
