import codecs
import numpy as np
from sklearn.preprocessing import normalize

# Load words
vocab = codecs.open("brown_vocab_100.txt")

# Create dictionary for indices
word_to_index = {}

# Create word_to_index
for word in vocab:
    # Add word to word_to_index if not included
    if not word.lower() in word_to_index.keys():
        word_to_index[word.lower().rstrip()] = len(word_to_index)
    
    prevWord = word

vocab.close()

# Create bigram model
condCounts = np.zeros((len(word_to_index),len(word_to_index)))

# Load sentences and tokenize
sents = codecs.open("brown_100.txt")

# Create word_to_index
for sentence in sents:
    words = sentence.lower().split(" ")[:-1]
    prevWord = words[0]
    for word in words[1:]:
        if prevWord in word_to_index.keys() and word in word_to_index.keys():
            wordIdx = word_to_index[word]          # Word along first axis
            condIdx = word_to_index[prevWord]      # Given along second axis
            condCounts[wordIdx, condIdx] = condCounts[wordIdx, condIdx] + 1
        
        prevWord = word

sents.close()

# Normalize for conditionals
bigram = normalize(condCounts, norm = 'l1', axis = 0)

# Tests
print("p(the|all):\t\t", bigram[word_to_index["the"], word_to_index["all"]])
print("p(jury|the):\t\t", bigram[word_to_index["jury"], word_to_index["the"]])
print("p(campaign|the):\t", bigram[word_to_index["campaign"], word_to_index["the"]])
print("p(calls|anonymous):\t", bigram[word_to_index["calls"], word_to_index["anonymous"]])

# Write probabilities to file
with open("bigram_probs.txt", 'w') as file:
    file.write(str(bigram[word_to_index["the"], word_to_index["all"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["jury"], word_to_index["the"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["campaign"], word_to_index["the"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["calls"], word_to_index["anonymous"]]))

# Define function for generating sentences
def GENERATE(word_index_dict, probs, model_type, max_words, start_word):
    returnSTR = ""
    index_word_dict = {v: k for k, v in word_index_dict.items()}
    num_words = 0

    #been passed a list of probabilities
    if model_type == "unigram":

        #using https://stackoverflow.com/questions/483666/python-reverse-invert-a-mapping
        while(True):
            wordIndex = np.random.choice(len(word_index_dict), 1, p=list(probs))
            word = index_word_dict[wordIndex[0]]
            returnSTR += word + " "
            num_words +=1
            if word == "</s>" or num_words == max_words:
                break

        return returnSTR

    #been passed a matrix of probabilities, where each row is the previous word. 
    if model_type == "bigram":
        returnSTR = start_word + " "
        prevWord = start_word
        while(True):
            wordIndex = np.random.choice(len(word_index_dict), 1, p=list(probs[word_index_dict[prevWord]]))
            word = index_word_dict[wordIndex[0]]
            returnSTR += word + " "
            num_words +=1
            prevWord = word
            if word == "</s>" or num_words == max_words:
                break

        return returnSTR

# Write sentences to file
with open("bigram_generation.txt", 'w') as file:
    for i in range(10):
        file.write(GENERATE(word_to_index, bigram.T, "bigram", 20, "the"))
        file.write("\n")
