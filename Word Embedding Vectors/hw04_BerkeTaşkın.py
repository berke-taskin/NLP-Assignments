import gensim.downloader
import random
import numpy as np

model = gensim.downloader.load("word2vec-google-news-300")

def replace_with_similar(sentence, indices):
    tokens = sentence.split()
    most_similar_dict = dict()

    # Generating Most Similar Words
    for i in indices:
        word = tokens[i]
        similar_words = model.most_similar(word, topn=5)
        most_similar_dict[word] = similar_words
        random_word = random.choice(similar_words)[0]
        tokens[i] = random_word

    # Forming New Sentence
    new_sentence = " ".join(tokens)

    return (new_sentence,  most_similar_dict)
    
def sentence_vector(sentence):
    vector_dict = dict()
    sentence_vec = []
    words = sentence.split()

    # Generating Vectors
    for word in words:
        if word in model:
            vector_dict[word] = model[word]
        else:
            vector_dict[word] = np.zeros(300)

    # Generating Sentence Vectors
    sentence_vec = np.sum(list(vector_dict.values()), axis=0) / len(vector_dict)

    return (vector_dict, sentence_vec)

def most_similar_sentences(file_path, query):
    # Opening File
    with open(file_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f.readlines()]

    # Query Vector
    query_vec = sentence_vector(query)[1]
    query_sentence_pairs = []

    # Sentence Vectors
    for sentence in sentences:
        sentence_vec = sentence_vector(sentence)[1]
        dot = np.dot(query_vec, sentence_vec)
        norm = np.linalg.norm(query_vec) * np.linalg.norm(sentence_vec)
        similarity = dot / norm
        query_sentence_pairs.append((sentence, similarity))

    # Sorting Pairs
    results = sorted(query_sentence_pairs, key=lambda x: x[1], reverse=True)

    return results

# For Testing Functions

# sentence = "I love AIN442 and BBM497 courses"
# indices = [1, 5]
# new_sentence, most_similar_dict = replace_with_similar(sentence, indices)
# print(most_similar_dict.keys(), end="\n\n")
# print(most_similar_dict["love"], end="\n\n")
# print(most_similar_dict["courses"], end="\n\n")
# print(new_sentence)

# vector_dict, sentence_vec = sentence_vector("This is a test sentence")
# print(vector_dict.keys(), end="\n\n")
# print(vector_dict["This"][:5], end="\n\n")
# print(vector_dict["a"][:5], end="\n\n")
# print(len(vector_dict["test"]))

# query_vec = sentence_vector("Which courses have you taken at Hacettepe University ?")[1]
# sentence_vec = sentence_vector("Students have the chance to gain practice with the homework given in lab classes of universities .")[1]
# print(query_vec[:5])
# print(sentence_vec[:5])

# file_path = "sentences.txt"
# query = "Which courses have you taken at Hacettepe University ?"
# results = most_similar_sentences(file_path, query)
# for sentence, score in results[:3]:
#     print(f"{score:.5f} -> {sentence}")

# Generating output.txt File
with open("output.txt", "w", encoding="utf-8") as f:
    sentence = "NLP is a fascinating field of study and I love learning about it"
    indices = [3, 4, 10]
    new_sentence, most_similar_dict = replace_with_similar(sentence, indices)
    
    print(most_similar_dict.keys(), end="\n\n", file=f)
    print(most_similar_dict["fascinating"], end="\n\n", file=f)
    print(most_similar_dict["field"], end="\n\n", file=f)
    print(most_similar_dict["learning"], end="\n\n", file=f)
    print(new_sentence, end="\n\n", file=f)
    
    print("----------------------------------------------------------------------------------------------------", end="\n\n", file=f)

    vector_dict, sentence_vec = sentence_vector("I am a student studying NLP at Hacettepe University")
    print(vector_dict.keys(), end="\n\n", file=f)
    print(vector_dict["I"][:5], end="\n\n", file=f)
    print(vector_dict["studying"][145:150], end="\n\n", file=f)
    print(vector_dict["Hacettepe"][295:], end="\n\n", file=f)

    print("----------------------------------------------------------------------------------------------------", end="\n\n", file=f)

    file_path = "sentences.txt"

    query1 = "Is swimming a good sport ?"
    results1 = most_similar_sentences(file_path, query1)
    for sentence, score in results1[:3]:
        print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)

    print("--------------------------------------------------", end="\n\n", file=f)

    query2 = "Does Turkey have good universities ?"
    results2 = most_similar_sentences(file_path, query2)
    for sentence, score in results2[:3]:
        print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)

    print("--------------------------------------------------", end="\n\n", file=f)

    query3 = "What happened to your backpack ?"
    results3 = most_similar_sentences(file_path, query3)
    for sentence, score in results3[:3]:
        print(f"{score:.5f} -> {sentence}", end="\n\n", file=f)