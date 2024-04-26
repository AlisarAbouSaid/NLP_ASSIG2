# Create partial bigram model, "all" is used for joint of all three, and "pair" for just the first two words
prevCounts = {
    ("in", "the"): 0,
    ("the", "jury"): 0,
    ("jury", "said"): 0,
    ("agriculture", "teacher"): 0
}

prevWords = {
    "past": ("in", "the"),
    "time": ("in", "the"),
    "said": ("the", "jury"),
    "recommended": ("the", "jury"),
    "that": ("jury", "said"),
    ",": ("agriculture", "teacher")
}

counts = {
    "past": 0,
    "time": 0,
    "said": 0,
    "recommended": 0,
    "that": 0,
    ",": 0
}

# Load sentences
sents = codecs.open("brown_100.txt")

# Get counts of conditionals
for sentence in sents:
    words = sentence.lower().split(" ")[:-1]
    
    if len(words) < 3: # skip if not enough words in sentence
        continue
    
    prevWord1 = words[0]
    prevWord2 = words[1]
    for word in words[2:]: # starting from 2 for trigram
        if (prevWord1, prevWord2) in prevCounts.keys():
            prevCounts[(prevWord1, prevWord2)] += 1
            if word in counts.keys() and prevWords[word] == (prevWord1, prevWord2):
                counts[word] += 1
        
        prevWord1 = prevWord2
        prevWord2 = word

# Calculate probabilities
probs = {
    "past | in, the": counts["past"] / prevCounts[("in", "the")],
    "time | in, the": counts["time"] / prevCounts[("in", "the")],
    "said | the, jury": counts["said"] / prevCounts[("the", "jury")],
    "recommended | the, jury": counts["recommended"] / prevCounts[("the", "jury")],
    "that | jury, said": counts["that"] / prevCounts[("jury", "said")],
    ", | agriculture, teacher": counts[","] / prevCounts[("agriculture", "teacher")]
}

alpha = 0.1
smoothCounts = {key: value + alpha for key, value in counts.items()}
smoothPrevCounts = {key: value + alpha * len(word_to_index) for key, value in prevCounts.items()}
smoothProbs = {
    "past | in, the": smoothCounts["past"] / smoothPrevCounts[("in", "the")],
    "time | in, the": smoothCounts["time"] / smoothPrevCounts[("in", "the")],
    "said | the, jury": smoothCounts["said"] / smoothPrevCounts[("the", "jury")],
    "recommended | the, jury": smoothCounts["recommended"] / smoothPrevCounts[("the", "jury")],
    "that | jury, said": smoothCounts["that"] / smoothPrevCounts[("jury", "said")],
    ", | agriculture, teacher": smoothCounts[","] / smoothPrevCounts[("agriculture", "teacher")]
}

# Print results
print("Unsmoothed trigram probabilities:")
print(*["p(" + key + ") = " + str(value) for key, value in probs.items()], sep = '\n', end = "\n\n")

print("Smoothed trigram probabilities:")
print(*["p(" + key + ") = " + str(value) for key, value in smoothProbs.items()], sep = '\n')