#!/usr/bin/env python3

"""
NLP A2: N-Gram Language Models

@author: Klinton Bicknell, Harry Eldridge, Nathan Schneider, Lucia Donatelli, Alexander Koller

DO NOT SHARE/DISTRIBUTE SOLUTIONS WITHOUT THE INSTRUCTOR'S PERMISSION
"""

import numpy as np
from generate import GENERATE

output_file_path = "C:/Users/alisa/Downloads/A2_2024 (2)/A2_2024/word_to_index_100.txt"
vocab_file_path = 'C:/Users/alisa/Downloads/A2_2024 (2)/A2_2024/brown_vocab_100.txt'
vocab = open("brown_vocab_100.txt")

# load the indices dictionary

word_index_dict = {}
with open(vocab_file_path, 'r') as file:
    for index, line in enumerate(file):
        word = line.strip()
        word_index_dict[word] = index

    # TODO: import part 1 code to build dictionary

f = open("brown_100.txt")

counts = np.zeros(len(word_index_dict))  # TODO: initialize counts to a zero vector
print(counts)
# TODO: iterate through file and update counts
with open("brown_100.txt", 'r') as file:
    for sentence in file:
        print(sentence)
        # Split the sentence into words and convert to lowercase
        words = sentence.lower().split(" ")
        # Increment counts for each word
        for word in words:
            if word in word_index_dict:
                print(word)
                # if word in word_index_dict:
                # print(word)
                index = word_index_dict[word]
                counts[index] = counts[index] + 1

# Print the counts vector
print(counts)
print(np.sum(counts))
# TODO: normalize and writeout counts.
# Normalize counts vector to probabilities
# probs = counts / np.sum(counts)
total_word_count = np.sum(counts)
probs = counts / total_word_count

# Initialize joint probability of the sentence
# sentprob = 1
perplexity_file_path = "perplexities.txt"
with open(perplexity_file_path, 'w') as perplexity_file:
    # Iterate through each sentence in the toy corpus
    # Open the output file for writing unigram probabilities
 output_file_path = "unigram_eval.txt"
 with open(output_file_path, 'w') as output_file:
    with open("toy_corpus.txt", 'r') as corpus_file:
        for sentence in corpus_file:
            # Split the sentence into words and convert to lowercase
            sentprob = 1
            words = sentence.lower().split(" ")
            # Calculate the joint probability of all the words in the sentence
            for word in words:
                # Check if the word exists in the vocabulary
                if word in word_index_dict:
                    # Calculate the probability of the word using the unigram model
                    # Update the joint probability of the sentence
                    sentprob *= probs[word_index_dict[word]]
            output_file.write(str(sentprob) + '\n')
            # Print the joint probability of the sentence
            print("Joint probability of the sentence:", sentprob)
            # Calculate the length of the sentence in words
            sent_len = len(words)-1
            print("sent_len",sent_len)
            # Calculate perplexity
            perplexity = 1/(pow(sentprob, 1/sent_len))
            # Write the perplexity of the sentence to the output file
            perplexity_file.write(str(perplexity) + '\n')
