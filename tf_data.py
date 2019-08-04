import tensorflow as tf
import numpy as np
tf.enable_eager_execution()  # 可以对tf debug


############## 创建  Dataset
#data = tf.random_uniform([4, 10])
#data1 = [[1, 2, 3, 4, 5]]
data = np.array([[1, 2, 3, 4, 5]])
dataset1 = tf.data.Dataset.from_tensor_slices(data)
#print(dataset.output_classes)
#print(dataset.output_shapes)  # ()
#print(dataset.output_types)
dataset2 = tf.data.Dataset.from_tensors(data)
dataset3 = tf.data.Dataset.zip((dataset1, dataset2))  #生成dataset1, dataset
iterator = dataset3.make_initializable_iterator()

iterator.initializer
next1, (next2, next3) = iterator.get_next()
#print(dataset.output_classes)
#print(dataset.output_shapes)  # (5,)
#print(dataset.output_types)

# 读取 npy 数据
with np.load("/var/data/training_data.npy") as data:
  features = data["features"]
  labels = data["labels"]
dataset = tf.data.Dataset.from_tensor_slices((features, labels))

features_placeholder = tf.placeholder(features.dtype, features.shape)
labels_placeholder = tf.placeholder(labels.dtype, labels.shape)

dataset = tf.data.Dataset.from_tensor_slices((features_placeholder, labels_placeholder))
# [Other transformations on `dataset`...]
dataset = dataset.map(function)
iterator = dataset.make_initializable_iterator()
sess = tf.Session()
sess.run(iterator.initializer, feed_dict={features_placeholder: features,
                                          labels_placeholder: labels})

# 读取TFRecord
#filenames = tf.placeholder(tf.string, shape=[None])
filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]

dataset = tf.data.TFRecordDataset(filenames)
dataset = dataset.map(...)  # Parse the record into tensors.
dataset = dataset.repeat()  # Repeat the input indefinitely.
dataset = dataset.batch(32)
#iterator = dataset.make_initializable_iterator()
iterator = dataset.make_one_shot_iterator()
training_filenames = ["/var/data/file1.tfrecord", "/var/data/file2.tfrecord"]
sess.run(iterator.initializer, feed_dict={filenames: training_filenames})



# 消耗文本数据
filenames = ["/var/data/file1.txt", "/var/data/file2.txt"]
dataset = tf.data.TextLineDataset(filenames)
dataset = dataset.flat_map(
    lambda filename: (
        tf.data.TextLineDataset(filename)
        .skip(1)
        .filter(lambda line: tf.not_equal(tf.substr(line, 0, 1), "#"))))


# 消耗csv数据
filenames = ["/var/data/file1.csv", "/var/data/file2.csv"]
record_defaults = [tf.float32] * 8   # Eight required float columns
# 可以分别使用 header 和 select_cols 参数移除这些行和字段。
dataset = tf.contrib.data.CsvDataset(filenames, record_defaults, header=True, select_cols=[2, 4])





"""
map  加工数据
"""
dataset1 = dataset1.map(lambda x: x+1) #  元素结构决定了函数的参数

dataset2 = dataset2.map(lambda x, y: x+y)

# Note: Argument destructuring is not available in Python 3.
# dataset3 = dataset3.filter(lambda x, (y, z): ...)

"""
获取Dataset中的数据
"""
dataset = dataset1.make_one_shot_iterator()# tf.data.Iterator 构建迭代器对象
while True:
    print(dataset.get_next())

