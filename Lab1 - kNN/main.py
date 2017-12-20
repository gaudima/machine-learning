from operator import itemgetter
import math
import matplotlib.pyplot as plt
import matplotlib.markers as mrk
from random import shuffle


def read_data(file):
    chips = []
    with open(file) as f:
        for chip in f:
            chip = chip.split(',')
            chips.append({'x': float(chip[0]), 'y': float(chip[1]), 'class': int(chip[2])})
    return chips


def split_for_cross_validation(data, step, steps):
    test_set_len = int(math.ceil(len(data) / steps))
    train_set = []
    test_set = []
    for i in range(len(data)):
        if test_set_len * step <= i < test_set_len * (step + 1):
            test_set.append(data[i])
        else:
            train_set.append(data[i])
    return train_set, test_set


def split_for_cross_validation2(data, step, steps):
    # test_set_len = len(data) // steps
    train_set = []
    test_set = []
    for i in range(len(data)):
        if i % steps == step:
            test_set.append(data[i])
        else:
            train_set.append(data[i])
    print(len(train_set), len(test_set))
    return train_set, test_set


def e_dist(a, b):
    return math.sqrt((a['x'] - b['x']) ** 2 + (a['y'] - b['y']) ** 2)


def cos_dist(a, b):
    zero = {'x': 0, 'y': 0}
    return (a['x'] * b['x'] + a['y'] * b['y']) / (e_dist(zero, a) * e_dist(zero, b))


def r_dist(a, b):
    zero = {'x': 0, 'y': 0}
    return math.fabs(e_dist(zero, a) - e_dist(zero, b))


def no_weight(a, b):
    return 1


def rev_exp_dist(a, b):
    return math.exp(-6 * e_dist(a, b))


def knn(train, test, k, dist, weight):
    labels = [{'x': d['x'], 'y': d['y'], 'class': d['class']} for d in test]
    for point_index in range(len(test)):
        point = test[point_index]
        test_dists = [{
            'x': train_point['x'],
            'y': train_point['y'],
            'class': train_point['class'],
            'd': dist(point, train_point)
        } for train_point in train]
        class_weights = [0] * 2
        test_dists = sorted(test_dists, key=itemgetter('d'))
        for d in test_dists[0:k]:
            class_weights[d['class']] += weight(point, d)
        max_w = -1
        max_i = -1
        for i in range(len(class_weights)):
            if max_w < class_weights[i]:
                max_w = class_weights[i]
                max_i = i
        labels[point_index]['class'] = max_i
    return labels


def get_labels(data):
    return [d['class'] for d in data]


def compute_rates(test, labeled):
    tp = 0  # is 1 classified as 1
    fp = 0  # is 0 classified as 1
    tn = 0  # is 0 classified as 0
    fn = 0  # is 1 classified as 0
    for i in range(len(test)):
        if test[i]['class'] == 1:
            if test[i]['class'] == labeled[i]['class']:
                tp += 1
            else:
                fn += 1
        else:
            if test[i]['class'] == labeled[i]['class']:
                tn += 1
            else:
                fp += 1

    return tp, fp, tn, fn


def compute_f1(tp, fp, fn):
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    return 2 * precision * recall / (precision + recall)


def plot_data(data, data2, colors, colors2):
    x = [d['x'] for d in data]
    y = [d['y'] for d in data]
    c = [colors[d['class']] for d in data]
    plt.scatter(x, y, c=c)

    xc = [d['x'] for d in data2]
    yc = [d['y'] for d in data2]
    cc = [colors2[d['class']] for d in data2]
    plt.scatter(xc, yc, c=cc, marker=mrk.MarkerStyle("+"))

    plt.show()


if __name__ == '__main__':
    data = read_data('chips.txt')
    # shuffle(data)
    steps = 10
    tp_all = 0
    fp_all = 0
    fn_all = 0
    labeled_all = []
    for step in range(steps):
        train, test = split_for_cross_validation2(data, step, steps)
        labeled = knn(train, test, 3, e_dist, rev_exp_dist)
        labeled_all += labeled
        print(get_labels(test))
        print(get_labels(labeled))
        print('----')
        tp, fp, tn, fn = compute_rates(test, labeled)
        tp_all += tp
        fp_all += fp
        fn_all += fn
    print(compute_f1(tp_all, fp_all, fn_all))
    plot_data(data, labeled_all, ['#ff0000', '#00ff00'], ['#ff0000', '#00ff00'])
