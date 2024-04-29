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

# Create bigram model
condCounts = np.zeros((len(word_to_index), len(word_to_index)))

# Load sentences and tokenize
sents = codecs.open("brown_100.txt")

# Create word_to_index
for sentence in sents:
    words = sentence.lower().split(" ")[:-1]
    prevWord = words[0]
    for word in words[1:]:
        if prevWord in word_to_index.keys() and word in word_to_index.keys():
            wordIdx = word_to_index[word]  # Word along first axis
            condIdx = word_to_index[prevWord]  # Given along second axis
            condCounts[wordIdx, condIdx] = condCounts[wordIdx, condIdx] + 1

        prevWord = word

# Normalize for conditionals
bigram = normalize(condCounts, norm='l1', axis=0)

# Tests
print("p(the|all):\t\t", bigram[word_to_index["the"], word_to_index["all"]])
print("p(jury|the):\t\t", bigram[word_to_index["jury"], word_to_index["the"]])
print("p(campaign|the):\t", bigram[word_to_index["campaign"], word_to_index["the"]])
print("p(calls|anonymous):\t", bigram[word_to_index["calls"], word_to_index["anonymous"]])

# Write file
with open("bigram_probs.txt", 'w') as file:
    file.write(str(bigram[word_to_index["the"], word_to_index["all"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["jury"], word_to_index["the"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["campaign"], word_to_index["the"]]))
    file.write("\n")
    file.write(str(bigram[word_to_index["calls"], word_to_index["anonymous"]]))

# Initialize joint probability of the sentence
# sentprob = 1
# Open the output file for writing bigram perplexities

bigram_output_file_path ="bigram_eval.txt"
with open(bigram_output_file_path, 'w') as bigram_output_file:
    # Iterate through each sentence in the toy corpus
    with open("toy_corpus.txt", 'r') as corpus_file:
        for sentence in corpus_file:
            # Split the sentence into words and convert to lowercase
            words = sentence.lower().split(" ")[:-1]
            # Initialize the joint probability of the sentence
            sentprob = 1
            num_bigrams = 0
            prevWord = words[0]
            # Calculate the joint probability of all the bigrams in the sentence
            for word in words[1:]:
                if prevWord in word_to_index.keys() and word in word_to_index.keys():
                    bigram_prob = bigram[word_to_index[word], word_to_index[prevWord]]
                    sentprob *= bigram_prob
                    num_bigrams += 1
                prevWord = word
            print('num_bigrams', num_bigrams)
            # Calculate perplexity
            perplexity = 1 / pow(sentprob, 1 / num_bigrams)
            # Write the perplexity of the sentence to the output file
            bigram_output_file.write(str(perplexity) + '\n')
