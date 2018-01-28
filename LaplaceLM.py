import re
from nltk import sent_tokenize, ngrams
from math import log


class LaplaceLM:
    
    def __init__(self, corpus_file, n=1, lim=4000):
        self.n=n
        self.lim=lim
        self.corpus = self.preprocess(self._read_corpus_file(corpus_file))
        bag = self.train()
        #self.test(bag)
        #self.score(bag)
        self.predict(["Of the", "I would like to"], bag)
        #self.print_bag(bag)
        
        
    def _read_corpus_file(self,path):  
        file = open(path, "r",encoding="utf8")
        return list(file)[:self.lim]
    
    def preprocess(self,x):
        corpus = []
        for text in x:
            sentences = sent_tokenize(text)
            for sent in sentences:
                sent = re.sub(r'[.,\/#?!$%\^&\*;:{}=\-_`~()\'\"\n]', '', sent.lower())
                start = ""
                for i in range(self.n-1):
                    start+= "$START{}$ ".format(i+1)
                sent= start+ sent+ " $END$"
                #sent= start+ sent+ (self.n>1)*" $END$"
                corpus+= [sent]
        corpus = " ".join(corpus)  
        return corpus        

    def train(self):
        bag_of_words = {}
        for k in set(self.corpus.split()):
            bag_of_words[k] = self.corpus.count(k)
        return bag_of_words    
    
    def print_bag(self):
        for key in sorted(bag_of_words.keys()):
            print("%s: %s" % (key, bag_of_words[key]))
            
    def prob(self, s, smoothing=0, voc=0):  
        #if (s==("$START1$",) or s==("$START2$",) or s==("$END$",) or s==("$START1$","$START2$")):
        #    return 1
        return ((self.n_count(s) + smoothing) / (self.d_count(s)+voc))

    def n_count(self, s):
        return self.corpus.count(" ".join(s[:len(s)]))

    def d_count(self, s):
        if(len(s)==0):
            return len(s.split())
        else:
            return self.corpus.count(" ".join(s[:len(s)-1]))   
     
    def test(self, bag):
        #print("skataaaaaaaaaaaaaaaaaaaaaaaaaaaaa")
        test = ["He please god football.", "He plays god football.", "He plays good football.", "He players good football.", "He pleases god ball.", "Of the rules.", "Rules the of."]
        #test = ["he please god football", "he plays god football", "he plays good football", "he players good football", "he pleases god ball", "of the rules"]
        for sec in test:
            print(sec)
            sec = self.preprocess([sec])
            
            print("\t{}-grams:".format(self.n))
            sum =0
            for s in ngrams(sec.split(" "),self.n):
                #for s in ngrams((["$START$"] + sec.split(" ") + ["$END$"]),n):
                p = self.prob(s, smoothing=1, voc=len(bag.keys()))
                sum+=log(p)
                print("\t\t",s, p)
            print("\tProbability is {}\n".format(sum)) 
            
    def score(self, bag):
        #test = ["Mr President, very briefly on a point of order regarding the texts adopted yesterday.", "During the debate, I was watching what exactly was written down regarding the vote for the Murphy report on late payment.", "Would you please rectify this?"]
        test = ["They have been"]
        entropy=0
        N = 0
        for sec in test:
            sec = self.preprocess([sec])
            for s in ngrams(sec.split(" "),self.n):
                p = self.prob(s, smoothing=1, voc=len(bag.keys()))
                entropy+=log(p)
                N+= 1
        entropy= -entropy/N        
        print("Entropy is {}\n".format(entropy))    

    def predict(self, preds, bag, ret=5):
        #test = ["he please god football", "he plays god football", "he plays good football", "he players good football", "he pleases god ball", "of the rules"]
        for seq in preds:
            voc = []
            seq = self.preprocess([seq])
            print(seq)
            print("\t{}-grams:".format(self.n))
    
            for word in bag.keys():
                nseq = re.sub("\$END\$", word, seq)
                nseq = (nseq.split())[-self.n:]
                p = self.prob(nseq, smoothing=1, voc=len(bag.keys())) *100
                voc+= [(" ".join(nseq),p)]
            voc = sorted(voc, key= lambda tup: tup[1])
            for (word,p) in voc[-ret:]:
                print("\t\t{}: {:4.2f}%".format(word,p))
          

for n in range(1,5):        
    lm = LaplaceLM("C:/Users/zeus8/Desktop/europarl-v7.el-en.en",n)        