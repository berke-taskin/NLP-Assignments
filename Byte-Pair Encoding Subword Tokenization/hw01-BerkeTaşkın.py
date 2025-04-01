"""
Assignment 1 - Code template for AIN442/BBM497

@author: İsmail Furkan Atasoy
"""

import re

def initialVocabulary():
    
    # You can use this function to create the initial vocabulary.
    
    return list("abcçdefgğhıijklmnoöprsştuüvyzwxq"+
                "ABCÇDEFGĞHIİJKLMNOÖPRSŞTUÜVYZWXQ"+
                "0123456789"+" "+
                "!'^#+$%&/{([)]=}*?\\_-<>|.:´,;`@€¨~\"é")

def bpeCorpus(corpus, maxMergeCount=10):     

    # TO DO
    # You can refer to Example 1, 2 and 3 for more details.

    # Token Learner from String
    (Merges, Vocabulary, TokenizedCorpus) = tokenLearner(corpus, maxMergeCount)

    return (Merges, Vocabulary, TokenizedCorpus) # Should return (Merges, Vocabulary, TokenizedCorpus)

def bpeFN(fileName, maxMergeCount=10):

    # TO DO
    # You can refer to Example 4 and 5 for more details.

    # Token Learner from File
    with open(fileName, 'r', encoding='utf-8') as infn:
        corpus = infn.read()
        (Merges, Vocabulary, TokenizedCorpus) = tokenLearner(corpus, maxMergeCount)
        
    return (Merges, Vocabulary, TokenizedCorpus) # Should return (Merges, Vocabulary, TokenizedCorpus)

def bpeTokenize(str, merges):

    # TO DO
    # You can refer to Example 6, 7 and 8 for more details.

    # Token Segmenter
    tokens = re.findall(r"\S+", str)
    tokenizedStr = [[" "] + list(token) + ["_"] for token in tokens]
    
    for merge in merges:
        for subwordToken in tokenizedStr:
            for i in range(len(subwordToken)-1):
                if i > len(subwordToken)-2:
                        break;
                if subwordToken[i] == merge[0][0] and subwordToken[i+1] == merge[0][1]:
                    subwordToken[i] = merge[0][0]+merge[0][1]
                    del subwordToken[i+1]

    return tokenizedStr # Should return the tokenized string as a list

def bpeFNToFile(infn, maxMergeCount=10, outfn="output.txt"):
    
    # Please don't change this function. 
    # After completing all the functions above, call this function with the sample input "hw01_bilgisayar.txt".
    # The content of your output files must match the sample outputs exactly.
    # You can refer to "Example Output Files" section in the assignment document for more details.
    
    (Merges,Vocabulary,TokenizedCorpus)=bpeFN(infn, maxMergeCount)
    outfile = open(outfn,"w",encoding='utf-8')
    outfile.write("Merges:\n")
    outfile.write(str(Merges))
    outfile.write("\n\nVocabulary:\n")
    outfile.write(str(Vocabulary))
    outfile.write("\n\nTokenizedCorpus:\n")
    outfile.write(str(TokenizedCorpus))
    outfile.close()

def tokenLearner(corpus, count):

    # Additional function to implement token learning algorithm.

    Merges = []
    Vocabulary = initialVocabulary()
    tokens = re.findall(r"[\S]+", corpus)
    TokenizedCorpus = [[" "] + list(token) + ["_"] for token in tokens]

    for x in range(count):
        pairDict = {}
        for subwordToken in TokenizedCorpus:
            for i in range(len(subwordToken)-1):
                pair = (subwordToken[i], subwordToken[i+1])
                if pair in pairDict:
                    pairDict[pair] +=1
                else:
                    pairDict[pair] = 1
        if len(pairDict) != 0:
            highestCount = min(pairDict, key = lambda pair : (-pairDict[pair], pair))
            Merges.append((highestCount, pairDict[highestCount]))
            merged = highestCount[0]+highestCount[1]
            Vocabulary.append(merged)

            for subwordToken in TokenizedCorpus:
                for i in range(len(subwordToken)-1):
                    if i > len(subwordToken)-2:
                        break;
                    if subwordToken[i] == highestCount[0] and subwordToken[i+1] == highestCount[1]:
                        subwordToken[i] = merged
                        del subwordToken[i+1]
        else:
            break;

    return (Merges, Vocabulary, TokenizedCorpus)

bpeFNToFile("hw01_bilgisayar.txt", 1000, "hw01-output1.txt")
bpeFNToFile("hw01_bilgisayar.txt", 200, "hw01-output2.txt")