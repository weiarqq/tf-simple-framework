import os
import tensorflow as tf
import numpy as np
from tensorboard import summary as summary_lib
from input_functions import train_input_fn, eval_input_fn
from models import lstm_model_fn


tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

model_dir = 'model'
x_train = []
y_train = []
x_eval = []
y_eval = []
x_test = []
y_test = []



embedding_matrix = []
params = {'embedding_initializer': embedding_matrix}


lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn,
                                         model_dir=os.path.join(model_dir, 'lstm_pretrained'),
                                         params=params)

lstm_classifier.train(input_fn=train_input_fn(), steps=25000)
eval_results = lstm_classifier.evaluate(input_fn=eval_input_fn)

predictions = np.array([p['logistic'][0] for p in lstm_classifier.predict(input_fn=eval_input_fn())])

tf.reset_default_graph()
pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool),
                          num_thresholds=21)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(lstm_classifier.model_dir, 'eval'), sess.graph)
    writer.add_summary(sess.run(pr), global_step=0)
    writer.close()






def print_predictions(x, label):

    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x, "label": label}, shuffle=False)
    predictions = {}
    for path, classifier in lstm_classifier.items():
        predictions[path] = [p['logistic'][0] for p in classifier.predict(input_fn=predict_input_fn)]



print_predictions([
    'I really liked the movie!',
    'Hated every second of it...'])

