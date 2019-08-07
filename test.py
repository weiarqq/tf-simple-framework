import tensorflow as tf
import numpy as np
tf.enable_eager_execution()

def input():
    data1 = np.array([[111, 22, 33, 44, 55],[1111, 22, 33, 44, 55],[11111, 22, 33, 44, 55]])
    dataset1 = tf.data.Dataset.from_tensor_slices(data1)

    data2 = np.array([[21, 2, 3, 4, 5],[221, 2, 3, 4, 5],[2221, 2, 3, 4, 5]])
    dataset2 = tf.data.Dataset.from_tensor_slices(data2)

    dataset = tf.data.experimental.sample_from_datasets([dataset1, dataset2], weights=[0.1, 0.2])
    dataset = dataset.make_one_shot_iterator()
    d = dataset.get_next()
    return d
i = 0
while i < 6:
    print(input())
    i += 1
