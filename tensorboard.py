import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data
max_steps = 1000
learing_rate = 0.001
dropout = 0.9
data_dir = './tmp/mnist'
log_dir = './tmp/logs'

mnist = input_data.read_data_sets(data_dir, one_hot=True)
sess = tf.InteractiveSession()

with tf.name_scope('input'):
    x = tf.placeholder(shape=[None, 784], dtype=tf.float32, name='x_input')
    y_ = tf.placeholder(shape=[None, 10], dtype=tf.float32, name='y_input')

with tf.name_scope('input_reshape'):
    image_shaped_input = tf.reshape(x, shape=[-1, 28, 28, 1])
    tf.summary.image('input', image_shaped_input, 10) # 记录训练数据 10个

# 日志 记录
def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var-mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var) # 记录变量var的直方图数据

# 定义隐藏层
def nn_layer(input_tensor, input_dim, out_dim, layer_name, act=tf.nn.relu):
    with tf.name_scope(layer_name):
        with tf.name_scope('weights'):
            W = tf.Variable(tf.truncated_normal(shape=[input_dim, out_dim], stddev=0.1))
            variable_summaries(W)
        with tf.name_scope('biases'):
            b = tf.Variable(tf.constant(0.1, shape=[out_dim]))
            variable_summaries(b)

        with tf.name_scope('wx_plus_b'):
            z = tf.add(tf.matmul(input_tensor, W), b)
            variable_summaries(z)
        activation = act(z, name='activation')

        tf.summary.histogram('activation', activation)  # 记录结果直方图
        return activation

hidden1 = nn_layer(x, 784, 500, 'layer1')

with tf.name_scope('dropout'):
    keep_prob = tf.placeholder(tf.float32)
    tf.summary.scalar('keep_prob', keep_prob)
    dropped = tf.nn.dropout(hidden1, keep_prob)

y = nn_layer(dropped, 500, 10, 'layer2', act=tf.identity)

with tf.name_scope('cross_entropy'):
    diff = tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)
    with tf.name_scope('total'):
        cross_entry = tf.reduce_mean(diff)
    tf.summary.scalar('cross_entropy', cross_entry)

with tf.name_scope('train'):
    opt = tf.train.AdamOptimizer(learing_rate).minimize(cross_entry)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    tf.summary.scalar('accuracy', accuracy)

merged = tf.summary.merge_all() # 获取所有汇总操作
train_writer = tf.summary.FileWriter(log_dir+'/train', sess.graph) #存放日志的文件  以及计算图
test_writer = tf.summary.FileWriter(log_dir+'/test')
init = tf.global_variables_initializer()
init.run()

def feed_dict(train):
    if train:
        xs, ys = mnist.train.next_batch(100)
        k = dropout
    else:
        xs, ys = mnist.test.images, mnist.test.labels
        k = 1.0

    return {x:xs, y_:ys, keep_prob:k}

saver = tf.train.Saver()

for i in range(max_steps):
    if i % 10 == 0:
        summary, acc = sess.run([merged, accuracy], feed_dict=feed_dict(False))
        test_writer.add_summary(summary, i)
        print('accuracy at step %s: %s'%(i, acc))
    else:
        if i % 100 == 99:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            # options=run_options, run_metadata=run_metadata 记录运行时间和内存大小
            summary, _ = sess.run([merged, opt], feed_dict=feed_dict(True), options=run_options, run_metadata=run_metadata)
            train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
            train_writer.add_summary(summary, i)
            saver.save(sess, log_dir+'model.ckpt', i)
            print('Adding run metadata for', i)
        else:
            summary, _ = sess.run([merged, opt], feed_dict=feed_dict(True))
            train_writer.add_summary(summary, i)

train_writer.close()
test_writer.close()