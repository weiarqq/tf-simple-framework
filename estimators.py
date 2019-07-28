import os
import string
import tempfile
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.datasets import imdb
from tensorflow.python.keras.preprocessing import sequence
from tensorboard import summary as summary_lib
from input_functions import train_input_fn, eval_input_fn
from utils import load_glove_embeddings, load_data
from models import cnn_model_fn, lstm_model_fn


tf.logging.set_verbosity(tf.logging.INFO)
print(tf.__version__)

vocab_size = 5000
sentence_size = 200
embedding_size = 50
#model_dir = tempfile.mkdtemp()
model_dir = 'model'
project_path = os.path.dirname(os.path.abspath(__file__))

(x_train_variable, y_train), (x_test_variable, y_test) = imdb.load_data(path=os.path.join(project_path, 'data/imdb.npz'),
    num_words=vocab_size)
print(len(y_train), "train sequences")
print(len(y_test), "test sequences")

print("Pad sequences (samples x time)")
x_train = sequence.pad_sequences(x_train_variable,
                                 maxlen=sentence_size,
                                 padding='post',
                                 value=0)
x_test = sequence.pad_sequences(x_test_variable,
                                maxlen=sentence_size,
                                padding='post',
                                value=0)

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)
word_index = imdb.get_word_index(os.path.join(project_path, 'data/imdb_word_index.json'))
word_inverted_index = {v: k for k, v in word_index.items()}
# The first indexes in the map are reserved to represet things other than tokens
index_offset = 3
word_inverted_index[-1 - index_offset] = '_' # Padding at the end
word_inverted_index[ 1 - index_offset] = '>' # Start of the sentence
word_inverted_index[ 2 - index_offset] = '?' # OOV
word_inverted_index[ 3 - index_offset] = ''  # Un-used





x_len_train = np.array([min(len(x), sentence_size) for x in x_train_variable])
x_len_test = np.array([min(len(x), sentence_size) for x in x_test_variable])
embedding_matrix = load_glove_embeddings('data/glove.6B.50d.txt', word_index, vocab_size, embedding_size)
params = {'embedding_initializer': embedding_matrix}
lstm_classifier = tf.estimator.Estimator(model_fn=lstm_model_fn,
                                                   model_dir=os.path.join(model_dir, 'cnn_pretrained'),
                                                   params=params)
# Save a reference to the classifier to run predictions later
lstm_classifier.train(input_fn=train_input_fn(x_train, x_len_train, y_train, x_train_variable), steps=500)
eval_results = lstm_classifier.evaluate(input_fn=eval_input_fn(x_test, x_len_test, y_test))

predictions = np.array([p['logistic'][0] for p in lstm_classifier.predict(input_fn=eval_input_fn(x_test, x_len_test, y_test))])

tf.reset_default_graph()
pr = summary_lib.pr_curve('precision_recall', predictions=predictions, labels=y_test.astype(bool),
                          num_thresholds=21)
with tf.Session() as sess:
    writer = tf.summary.FileWriter(os.path.join(lstm_classifier.model_dir, 'eval'), sess.graph)
    writer.add_summary(sess.run(pr), global_step=0)
    writer.close()






def print_predictions(sentences):
    def text_to_index(sentence):
        # Remove punctuation characters except for the apostrophe
        translator = str.maketrans('', '', string.punctuation.replace("'", ''))
        tokens = sentence.translate(translator).lower().split()
        return np.array([1] + [word_index[t] + index_offset if t in word_index else 2 for t in tokens])

    indexes = [text_to_index(sentence) for sentence in sentences]
    x = sequence.pad_sequences(indexes,
                               maxlen=sentence_size,
                               padding='post',
                               value=0)
    label = np.array([min(len(x), sentence_size) for x in indexes])
    predict_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": x, "len": label}, shuffle=False)
    predictions = [p['logistic'][0] for p in lstm_classifier.predict(input_fn=predict_input_fn)]
    for idx, sentence in enumerate(sentences):
        print(sentence)
        print(str(predictions[idx]))



print_predictions([
    'I really liked the movie!',
    'Hated every second of it...'])

