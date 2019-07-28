import tensorflow as tf




def parser(x, length, y):
    features = {"x": x, "len": length}
    return features, y

def train_input_fn(x_train, x_len_train, y_train, x_train_variable):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_train, x_len_train, y_train))
        dataset = dataset.shuffle(buffer_size=len(x_train_variable))
        dataset = dataset.batch(100)
        dataset = dataset.map(parser)
        dataset = dataset.repeat()
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


def eval_input_fn(x_test, x_len_test, y_test):
    def input_fn():
        dataset = tf.data.Dataset.from_tensor_slices((x_test, x_len_test, y_test))
        dataset = dataset.batch(100)
        dataset = dataset.map(parser)
        iterator = dataset.make_one_shot_iterator()
        return iterator.get_next()
    return input_fn


######################################################################################
import tensorflow as tf
from tensorflow import data
import multiprocessing

MULTI_THREADING = True

SEQUENCE_LENGTH = 128
DEFAULTS = [[0.0] for i in range(0, SEQUENCE_LENGTH)]
VALUES_FEATURE_NAME = 'values'


def parse_csv_row(words, label):
    # sequence is a list of tensors

    input_data = tf.concat(words, axis=1)
    one_hot_label = tf.one_hot(label, 7, 1, 0, axis=1)

    return {VALUES_FEATURE_NAME: input_data}, one_hot_label

def csv_input_fn(files_name_pattern, mode=tf.estimator.ModeKeys.EVAL,
                 skip_header_lines=0,
                 num_epochs=1,
                 batch_size=20):

    file_names = tf.matching_files(files_name_pattern)
    dataset = data.TextLineDataset(filenames=file_names)
    dataset = dataset.skip(skip_header_lines)
    num_threads = multiprocessing.cpu_count() if MULTI_THREADING else 1
    if mode == tf.estimator.ModeKeys.TRAIN:
        dataset = dataset.shuffle(buffer_size=2 * batch_size + 1)

    dataset = dataset.batch(batch_size)
    dataset = dataset.map(lambda csv_row: parse_csv_row(csv_row),
                          num_parallel_calls=num_threads)

    # dataset = dataset.batch(batch_size) #??? very long time
    dataset = dataset.repeat(num_epochs)
    iterator = dataset.make_one_shot_iterator()

    features, target = iterator.get_next()
    return features, target



def file_based_input_fn_builder(input_file, seq_length, is_training, drop_remainder, batch_size):
    name_to_features = {
        "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "label_ids": tf.FixedLenFeature([seq_length], tf.int64),
        "not_nulls": tf.FixedLenFeature([seq_length], tf.int64)
    }

    def _decode_record(record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def input_fn():
        d = tf.data.TFRecordDataset(input_file)
        if is_training:
            d = d.repeat()
            d = d.shuffle(buffer_size=300)
        d = d.apply(tf.data.experimental.map_and_batch(lambda record: _decode_record(record, name_to_features),
                                                       batch_size=batch_size,
                                                       num_parallel_calls=8,  # 并行处理数据的CPU核心数量，不要大于你机器的核心数
                                                       drop_remainder=drop_remainder))
        d = d.prefetch(buffer_size=4)
        return d

    return input_fn