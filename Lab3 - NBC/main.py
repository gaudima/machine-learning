import os
import math
import numpy as np


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
                samples.append({"data": words, "is_spam": is_spam, "max_word": max(words), "file_name": file_name})
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


def classify(classifier, samples, blur_k, margin):
    spam_files, legit_files, all_files, unique_words, \
        spam_words, legit_words, spam_word_count, legit_word_count = classifier
    classified = [(False, 0)] * len(samples)
    for index, sample in enumerate(samples):
        spam_log = math.log(spam_files / all_files)
        legit_log = math.log(legit_files / all_files)
        for word in sample["data"]:
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
        try:
            exp = math.exp(spam_log - legit_log)
        except OverflowError:
            exp = float("inf")
        legit_probability = 1 / (1 + exp)
        if 1 - legit_probability >= margin:
        # if math.fabs(legit_log) / math.fabs(spam_log) > 1.1:
            classified[index] = (True, 1 - legit_probability)
        else:
            classified[index] = (False, 1 - legit_probability)
    return classified


def load_all_samples():
    samples = []
    for i in range(1, 11):
        samples.append(load_samples("pu1", i))
    return samples

samples = load_all_samples()
min_error = 100
min_blur_k = 0
min_margin = 0
for blur_k in range(0, 11):
    for margin in np.arange(0.1, 1.1, 0.1):
        cv_error = 0
        for cv_on in range(1, len(samples)):
            test_samples = samples[cv_on]
            train_samples = []
            for i in range(1, len(samples)):
                if i != cv_on:
                    train_samples.extend(samples[i])
            classified = classify(train(train_samples), test_samples, blur_k, margin)
            wrong_num = 0
            for idx, sample in enumerate(test_samples):
                if classified[idx][0] != sample["is_spam"]:
                    wrong_num += 1
            cv_error += wrong_num / len(test_samples)
        if min_error > cv_error:
            min_error = cv_error
            min_blur_k = blur_k
            min_margin = margin
        print("cross-validation error: {:.2f}% | blur_k = {} | margin = {:.1f}".format(cv_error / 10 * 100, blur_k, margin))
print("\nminimal error: {:.2f}% | blur_k = {} | margin = {:.1f}".format(min_error / 10 * 100, min_blur_k, min_margin))
