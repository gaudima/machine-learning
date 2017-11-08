import os
import math
import numpy as np
import matplotlib.pyplot as pyplt


def load_samples(path, group):
    if not path.endswith("/"):
        path += "/"
    path += "part" + str(group)
    samples = []
    for file_name in os.listdir(path):
        if file_name.endswith(".txt"):
            with open(os.path.join(path, file_name)) as file:
                data = file.read()
                is_spam = False
                if "spmsg" in file_name:
                    is_spam = True
                words = list(map(int, data.split()[1:]))
                samples.append({"data": words,
                                "topic_len": len(data.split("\n")[0].split()[1:]),
                                "is_spam": is_spam,
                                "max_word": max(words),
                                "file_name": file_name})
    return samples


def train(samples):
    all_files = len(samples)
    spam_files = 0
    legit_files = 0
    unique_words = 0
    spam_words = 0
    legit_words = 0
    spam_word_count = {}
    legit_word_count = {}
    for sample in samples:
        if sample["is_spam"]:
            spam_words += len(sample["data"])
            spam_files += 1
        else:
            legit_words += len(sample["data"])
            legit_files += 1
        unique_words = max(unique_words, sample["max_word"])
        for word in sample["data"]:
            if sample["is_spam"]:
                if word not in spam_word_count:
                    spam_word_count[word] = 1
                else:
                    spam_word_count[word] += 1
            else:
                if word not in legit_word_count:
                    legit_word_count[word] = 1
                else:
                    legit_word_count[word] += 1
    return spam_files, legit_files, all_files, unique_words, spam_words, legit_words, spam_word_count, legit_word_count


def classify(classifier, samples, blur_k, topic_k):
    spam_files, legit_files, all_files, unique_words, \
        spam_words, legit_words, spam_word_count, legit_word_count = classifier
    classified = [0] * len(samples)
    for index, sample in enumerate(samples):
        spam_log = math.log(spam_files / all_files)
        legit_log = math.log(legit_files / all_files)
        for idx, word in enumerate(sample["data"]):
            spam_word_count1 = blur_k
            if word in spam_word_count:
                spam_word_count1 += spam_word_count[word]
            legit_word_count1 = blur_k
            if word in legit_word_count:
                legit_word_count1 += legit_word_count[word]
            if spam_word_count1 > 0:
                spam_log += math.log(spam_word_count1 / (blur_k * unique_words + spam_words))
            if legit_word_count1 > 0:
                legit_log += math.log(legit_word_count1 / (blur_k * unique_words + legit_words))
            if idx < sample["topic_len"]:
                spam_log += math.log(topic_k)
                legit_log += math.log(topic_k)
            classified[index] = spam_log - legit_log
    return classified


def load_all_samples():
    samples = []
    for i in range(1, 11):
        samples.append(load_samples("pu1", i))
    return samples

samples = load_all_samples()

margin = +float('inf')
for cv_on in range(1, len(samples)):
    test_samples = samples[cv_on]
    train_samples = []
    for i in range(1, len(samples)):
        if i != cv_on:
            train_samples.extend(samples[i])
    classified = classify(train(train_samples), test_samples, 1, 1)
    for idx, sample in enumerate(test_samples):
        if not sample["is_spam"]:
           margin = min(margin, classified[idx])
print(margin)