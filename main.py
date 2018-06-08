import csv
import numpy as np
import collections

train_data = []
test_data = []

for i in range(10):
    train_file = open("digit/digit_train%d.csv" % i)
    train_f = csv.reader(train_file)

    for row in train_f:
        train_data.append([i, [float(s) for s in row]])

    test_file = open("digit/digit_test%d.csv" % i)
    test_f = csv.reader(test_file)

    for row in test_f:
        test_data.append([i, [float(s) for s in row]])


def get_nearest_label(data, train_data, k):
    distance_list = []
    for i in range(len(train_data)):
        distance = 0
        for j in range(len(train_data[i][1])):
            distance += (data[1][j] - train_data[i][1][j]) ** 2

        distance_list.append(distance)

    label_list = []
    min_index = 0
    for i in range(k):
        min_index = distance_list.index(min(distance_list))
        label_list.append(train_data[min_index][0])
        distance_list[min_index] += 10000

    counter = collections.Counter(label_list)
    sorted_counter = [(v, k) for k, v in counter.items()]
    sorted_counter.sort()
    estimated_label = sorted_counter[len(sorted_counter)-1][1]

    return estimated_label


def error_by_k():
    k_list = [1, 2, 3, 4, 5]
    t = 5
    train_data_slice_range = int(len(train_data) / t)
    np.random.shuffle(train_data)

    for i in range(len(k_list)):
        k = k_list[i]
        num_error = 0

        for j in range(t):
            excluded_train_data = train_data[train_data_slice_range*j:train_data_slice_range*(j+1)]
            sliced_train_data = train_data[:train_data_slice_range*j] + train_data[train_data_slice_range*(j+1):]

            for l in range(len(excluded_train_data)):
                label = get_nearest_label(excluded_train_data[l], sliced_train_data, k)

                if label != excluded_train_data[l][0]:
                    num_error += 1

        print("誤識別数 : %d" % num_error)


def main():
    estimate_matrix = np.zeros([10, 10])
    num_error = 0

    for i in range(len(test_data)):
        estimated_label = get_nearest_label(test_data[i], train_data, 1)
        if estimated_label != test_data[i][0]:
            num_error += 1

        estimate_matrix[test_data[i][0]][estimated_label] += 1

    print(estimate_matrix)
    print("誤識別数 : %d" % num_error)


if __name__ == '__main__':
    main()
