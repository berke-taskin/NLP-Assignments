# -*- coding: utf-8 -*-
"""
Created on Sat Mar  8 15:06:51 2025

@author: ilyas
"""

import math
import random
import re
import codecs


# ngramLM CLASS
class ngramLM:
    """Ngram Language Model Class"""
    
    # Create Empty ngramLM
    def __init__(self):
        self.numOfTokens = 0
        self.sizeOfVocab = 0
        self.numOfSentences = 0
        self.sentences = []
        # TO DO 
        self.vocabulary = dict()
        self.bigramsDict = dict()

    # INSTANCE METHODS
    def trainFromFile(self,fn):
        # TO DO
        with codecs.open(fn, "r", encoding="utf-8") as file:
            #Regex Patterns
            pattern = r"""(?x)  
                    (?:[A-ZÇĞIİÖŞÜ]\.)+              
                    | \d+(?:\.\d*)?(?:\'\w+)?   
                    | \w+(?:-\w+)*(?:\'\w+)?  
                    | \.\.\.  
                    | [][,;.?():_!#^+$%&><|/{()=}\"\'\\\"\`-] 
                    """
            #Reading lines from the file
            lines = file.readlines()
            #Iterating through lines
            for line in lines:
                tokenized = re.findall(pattern, line)
                if tokenized:
                    wordList = ['<s>']
                    for i, token in enumerate(tokenized):
                        #Converting "I" and "İ" to lower cases
                        for j in range(len(token)):
                            if token[j] == "I":
                                token = token[:j]+"ı"+token[j+1:]
                            elif token[j] == "İ":
                                token = token[:j]+"i"+token[j+1:]
                        token = token.lower()
                        wordList.append(token)
                        #End of the sentence or line
                        if token=="." or token=="?"  or token=="!" or i==len(tokenized)-1:
                            #Forming sentences
                            wordList.append('</s>')  
                            self.sentences.append(wordList)
                            self.numOfTokens += len(wordList)
                            #Adding tokens to vocabulary
                            for word in wordList:
                                if word not in self.vocabulary:
                                    self.vocabulary[word] = 1
                                else: 
                                    self.vocabulary[word] = self.vocabulary[word]+1
                            #Adding pair tokens to bigrams
                            for j in range(len(wordList)-1):
                                bigram = (wordList[j], wordList[j+1])
                                if bigram not in self.bigramsDict:
                                    self.bigramsDict[bigram] = 1
                                else: 
                                    self.bigramsDict[bigram] = self.bigramsDict[bigram]+1
                            wordList = ['<s>'] 
            #Sorting Vocabulary and Bigrams
            self.vocabulary = dict(sorted(self.vocabulary.items(), key = lambda item: (-item[1], item[0])))
            self.bigramsDict = dict(sorted(self.bigramsDict.items(), key = lambda item: (-item[1], item[0])))
            #Setting number of sentences variable
            self.numOfSentences = len(self.sentences)
            self.sizeOfVocab = len(self.vocabulary)
                         
    def vocab(self):
        # TO DO
        vocabList = []
        for word in self.vocabulary:
            vocabList.append((word, self.vocabulary[word]))
        return vocabList
    
    def bigrams(self):
        # TO DO
        bigramList = []
        for pair in self.bigramsDict:
            bigramList.append((pair, self.bigramsDict[pair]))
        return bigramList

    def unigramCount(self, word):
        # TO DO
        if word in self.vocabulary:
            return self.vocabulary[word]
        else:
            return 0

    def bigramCount(self, bigram):
        # TO DO
        if bigram in self.bigramsDict:
            return self.bigramsDict[bigram]
        else:
            return 0

    def unigramProb(self, word):
        # TO DO
        # returns unsmoothed unigram probability value
        if word in self.vocabulary:
            return self.vocabulary[word]/self.numOfTokens
        else:
            return 0

    def bigramProb(self, bigram):
        # TO DO
        # returns unsmoothed bigram probability value
        if bigram in self.bigramsDict:
            return self.bigramsDict[bigram]/self.vocabulary[bigram[0]]
        else:
            return 0

    def unigramProb_SmoothingUNK(self, word):
        # TO DO
        # returns smoothed unigram probability value
        if word in self.vocabulary:
            return (self.vocabulary[word]+1)/(self.numOfTokens+(self.sizeOfVocab+1))
        else:
            return (1)/(self.numOfTokens+(self.sizeOfVocab+1))

    def bigramProb_SmoothingUNK(self, bigram):
        # TO DO
        # returns smoothed bigram probability value
        if bigram in self.bigramsDict:
            return (self.bigramsDict[bigram]+1)/(self.vocabulary[bigram[0]]+(self.sizeOfVocab+1))
        else:
            if bigram[0] in self.vocabulary:
                return (1)/(self.vocabulary[bigram[0]]+(self.sizeOfVocab+1))
            else:
                return (1)/(self.sizeOfVocab+1)
    
    def sentenceProb(self,sent):
        # TO DO 
        # sent is a list of tokens
        # returns the probability of sent using smoothed bigram probability values
        if len(sent) > 1:
            logProb = 0
            for i in range(len(sent)-1):
                bigram = (sent[i], sent[i+1])
                logProb += math.log(self.bigramProb_SmoothingUNK(bigram))
            return math.exp(logProb)
        else:
            return self.unigramProb_SmoothingUNK(sent[0])
    
    def generateSentence(self,sent=["<s>"],maxFollowWords=1,maxWordsInSent=20):
        # TO DO 
        # sent is a list of tokens
        # returns the generated sentence (a list of tokens)
        limit = True
        # While word count in sentence is not 20
        while maxWordsInSent != 0:
            followingDict = dict()
            following = 0
            freq = 0
            # Iterating through possible bigrams
            for bigram in self.bigramsDict:
                if sent[-1] == bigram[0]:
                    freq += self.bigramsDict[bigram]
                    followingDict[bigram] = freq
                    following += 1
                if following >= maxFollowWords:
                    break;
            x = random.randint(1,freq)
            # Finding randomly chosen bigram
            for bigram in followingDict:
                if followingDict[bigram] >= x:
                    word = bigram[1]
                    break;
            sent.append(word)
            maxWordsInSent -= 1
            # If the end of line is reached without 20 words
            if word == '</s>':
                limit = False
                break;
        # Ending the sentence if word count has reached 20
        if limit:
            sent.append('</s>')
        return sent