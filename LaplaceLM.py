import re
from nltk import sent_tokenize, ngrams
from math import log
import random

class LaplaceLM:

    def __init__(self, corpus_file, n=1, start=0, end=4000, unknown_threshold=10):
        self.n = n
        self.corpus = self._preprocess(self._read_corpus_file(corpus_file, start, end))
        self.train_vocabulary = self._train(self.corpus,unknown_threshold)
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

    def _train(self, dataset, threshold):
        distinct_words = set(dataset.split())
        vocabulary = [word if dataset.count(word) > threshold else '*UNK*' for word in distinct_words]
        return set(vocabulary)

    def _get_probability(self, n_gram):
        c1 = self.corpus.count(' '.join(n_gram))
        c2 = self.corpus.count(' '.join(n_gram[:len(n_gram)-1]))
        return (c1 + 1) / (c2 + len(self.train_vocabulary))
    
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
        print("\t{}-grams:".format(self.n))
        for word in self.train_vocabulary:
            seq = re.sub("\*END\*", word, sequence)
            seq = (seq.split())[-self.n:]
            prob = self._get_probability(seq) * 100
            candidates += [(' '.join(seq), prob)]
        candidates = sorted(candidates, key= lambda tup: tup[1])
        for (word, prob) in candidates[-results:]:
            print("\t\t{}: {:4.2f}%".format(word, prob))



for n in range(2,5):
    lm = LaplaceLM("C:\europarl\europarl-v7.el-en.en",n)
    test_sequences = []
    test_set_sentence = lm.get_test_sentence()
    test_sequences.append(test_set_sentence)
    for i in range(0,4):
        test_sequences.append(lm.create_random_sentence(len(test_set_sentence.split())))
    #test_sequences = ["He please god football.", "He plays god football.", "He plays good football.", "He players good football.", "He pleases god ball.", "Of the rules.", "Rules the of."]
    predict_sequences = ["Of the", "I would like to"]

    for test_sequence in test_sequences:
        lm.test(test_sequence)
    
    
    #lm.eval_measures(lm.test_sentences)

    for predict_sequence in predict_sequences:
        lm.predict(predict_sequence)
