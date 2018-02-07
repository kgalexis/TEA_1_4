from LaplaceLM import LaplaceLM

class InterpolatedLM:

    def __init__(self, corpus, l, threshold=5):
        self.bigram_lm = LaplaceLM(corpus, 2, threshold)
        self.trigram_lm = LaplaceLM(corpus, 3, threshold)
        self.l = l

    def evaluate(self, corpus):
        entropy = 0
        N = 0
        for sentence in corpus:
            entropy += self.l * self.bigram_lm.test(sentence) + (1 - self.l) * self.trigram_lm.test(sentence)
            N += len(sentence.split())
        entropy = -entropy/N
        perplexity = 2**entropy
        print("Entropy is {}".format(entropy))   
        print("Perplexity is {}".format(perplexity))
        return
