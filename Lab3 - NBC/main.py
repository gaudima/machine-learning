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
                is_spam = 0
                if "spmsg" in file_name:
                    is_spam = 1
                words = list(map(int, data.split()[1:]))
                samples.append({"data": words,
                                "topic_len": len(data.split("\n")[0].split()[1:]),
                                "class": is_spam,
                                "max_word": max(words),
                                "file_name": file_name})
    return samples


def train(samples):
    class_size = [0, 0]
    word_count = {}
    word_count_in_class = [0, 0]
    unique_words = set()
    for sample in samples:
        class_size[sample["class"]] += 1
        unique_words.update(set(sample["data"]))
        for word in sample["data"]:
            if word not in word_count:
                word_count[word] = [0, 0]
            word_count[word][sample["class"]] += 1
            word_count_in_class[sample["class"]] += 1
    return class_size, word_count, word_count_in_class, len(unique_words)


def classify(classifier, sample, blur_k, margin):
    class_size, word_count, word_count_in_class, unique_words = classifier
    log_prob = [.0, .0]

    for idx, word in enumerate(sample["data"]):
        for i in range(2):
            word_count_blured = blur_k
            if word in word_count:
                word_count_blured += word_count[word][i]
            log_prob[i] -= math.log(word_count_blured / (blur_k * unique_words + word_count_in_class[i]))
    for i in range(2):
        log_prob[i] -= math.log(class_size[i]/(class_size[0] + class_size[1]))
        # if margin == -1:
        #     return 1 if log_prob[1] < log_prob[0] else 0, log_prob[0], log_prob[1]
        # else:
        return 1 if log_prob[0] - log_prob[1] > margin else 0, log_prob[0] - log_prob[1]


def load_all_samples():
    samples = []
    for i in range(1, 11):
        samples.append(load_samples("pu1", i))
    return samples


def avoid_zero_div(a, b):
    if b == 0:
        return 0.0
    return a / b


def calculate_score(classifier, test_data, blur_k, margin):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for sample in test_data:
        classified = classify(classifier, sample, blur_k, margin)
        class_id = sample["class"]
        deduced_id = classified[0]
        if class_id == deduced_id:
            if class_id == 1:
                tp += 1
            else:
                tn += 1
        else:
            if class_id == 1:
                fn += 1
            else:
                fp += 1
    tpr = avoid_zero_div(tp, tp + fn)
    # fpr = avoid_zero_div(fp, fp + tn)
    # tnr = avoid_zero_div(tn, tn + fp)
    # fnr = avoid_zero_div(fn, tp + fn)
    precision = avoid_zero_div(tp, tp + fp)
    recall = tpr
    return avoid_zero_div(2 * precision * recall, precision + recall), tp, fp, tn, fn

samples = load_all_samples()
f1_avg = 0
tp_all = 0
tn_all = 0
fn_all = 0
fp_all = 0
blur_k = 0.0000000000001
margin = float("-inf")
for cv_on in range(len(samples)):
    test_samples = samples[cv_on]
    train_samples = []
    for i in range(len(samples)):
        if i != cv_on:
            train_samples.extend(samples[i])
    classifier = train(train_samples)
    for sample in test_samples:
        if sample["class"] == 0:
            margin = max(margin, classify(classifier, sample, blur_k, 0)[1])
print("margin:", margin)

for cv_on in range(len(samples)):
    test_samples = samples[cv_on]
    train_samples = []
    for i in range(len(samples)):
        if i != cv_on:
            train_samples.extend(samples[i])
    classifier = train(train_samples)
    f1, tp, fp, tn, fn = calculate_score(classifier, test_samples, blur_k, margin)
    f1_avg += f1
    tp_all += tp
    tn_all += tn
    fp_all += fp
    fn_all += fn

tpr = tp_all / (tp_all + fn_all)
fpr = fp_all / (fp_all + tn_all)
tnr = tn_all / (tn_all + fp_all)
fnr = fn_all / (tp_all + fn_all)
prec = tp_all / (tp_all + fp_all)
rec = tp_all / (tp_all + fn_all)
f1 = 2 * prec * rec / (prec + rec)
print("f1:", f1)
print("avg: tpr:", tpr, "fnr:", fnr, "tnr:", tnr, "fpr:", fpr)
# print("avg: f1:", f1_avg / 10)
# print("avg: tpr:", tpr_avg / 10, "fnr:", fnr_avg / 10, "tnr:", tnr_avg / 10, "fpr:", fpr_avg / 10)
