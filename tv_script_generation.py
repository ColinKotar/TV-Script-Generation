import helper
import warnings
import operator
import numpy as np
import tensorflow as tf
import problem_unittests as tests
from collections import Counter
from distutils.version import LooseVersion
from tensorflow.contrib import seq2seq

data_dir = './data/simpsons/moes_tavern_lines.txt'
save_dir = './save'
text = helper.load_data(data_dir)
text = text[81:] # skip copyright line


def print_data(view_sentence_range=(0, 10)):
    """
    Print data to get an idea of what the dataset looks like
    :param view_sentence_range: number of sentences(lines) to be viewed
    """

    print('Dataset Stats')
    print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
    scenes = text.split('\n\n')
    print('Number of scenes: {}'.format(len(scenes)))
    sentence_count_scene = [scene.count('\n') for scene in scenes]
    print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

    sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
    print('Number of lines: {}'.format(len(sentences)))
    word_count_sentence = [len(sentence.split()) for sentence in sentences]
    print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

    print()
    print('The sentences {} to {}:'.format(*view_sentence_range))
    print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))


def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """

    # unique set of words
    words = set(text)

    # dictionary of id as key to word
    int_to_vocab = {i: word for i, word in enumerate(words)}

    # dictionary of word as key to id
    vocab_to_int = {word: i for i, word in int_to_vocab.items()}

    # return tuple of both dictionaries
    return vocab_to_int, int_to_vocab


def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    # return dictionary of punctuation as key to tokenization
    return {'.': '||Period||',
            ',': '||Comma||',
            '"': '||Quotation_Mark||',
            ';': '||Semicolon||',
            '!': '||Exclamation_Mark||',
            '?': '||Question_Mark||',
            '(': '||Left_Parentheses||',
            ')': '||Right_Parentheses||',
            '--': '||Dash||',
            '\n': '||Return||'}


def check_tf_gpu():
    # Check TensorFlow Version
    assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
    print('TensorFlow Version: {}'.format(tf.__version__))

    # Check for a GPU
    if not tf.test.gpu_device_name():
        warnings.warn('No GPU found. Please use a GPU to train your neural network.')
    else:
        print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))


def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """

    # create placeholders
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32)

    # return tuple of TF Placeholders
    return (inputs, targets, learning_rate)


def get_init_cell(batch_size, rnn_size):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """

    # my LSTMs that will be stacked in an RNN
    lstm_0 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_1 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_2 = tf.contrib.rnn.BasicLSTMCell(rnn_size)
    lstm_3 = tf.contrib.rnn.BasicLSTMCell(rnn_size)

    # my multi RNN cell
    rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm_0, lstm_1, lstm_2, lstm_3])

    # setting initial state with batch_size
    init_state = rnn_cell.zero_state(batch_size, tf.float32)
    init_state = tf.identity(init_state, name='initial_state')

    # return tuple with cell and initial state
    return (rnn_cell, init_state)


def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """

    domain = (-0.1, 0.1)

    # apply embedding using tf
    embedding = tf.Variable(tf.random_uniform((vocab_size, embed_dim), domain[0], domain[1]))
    embed = tf.nn.embedding_lookup(embedding, input_data)

    # return embedded input
    return embed


def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """

    # create an RNN
    outputs, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)

    # apply name
    final_state = tf.identity(final_state, name='final_state')

    # return tuple of outputs and final state
    return (outputs, final_state)


def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """

    # apply embedding
    embedding = get_embed(input_data, vocab_size, embed_dim)

    # build RNN
    outputs, final_state = build_rnn(cell, embedding)

    # apply fully connected layer
    logits = tf.contrib.layers.fully_connected(outputs, vocab_size,
                                               activation_fn=None,
                                               weights_initializer = tf.contrib.layers.xavier_initializer(),
                                               biases_initializer = tf.zeros_initializer())

    # return tuple of logits & final_state
    return (logits, final_state)


def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """

    # calculate ints_per_batch and num_batches
    ints_per_batch = batch_size * seq_length
    num_batches = len(int_text)//ints_per_batch

    # create inputs array from int_text
    inputs = np.array(int_text[:(num_batches * ints_per_batch)])

    # create targets array from int_text
    targets = np.array(int_text[1:(num_batches * ints_per_batch)+1])

    # set last value in targets to first value in inputs
    targets[-1] = inputs[0]

    # transform the inputs and targets arrays
    input_batch = inputs.reshape(batch_size, -1)
    target_batch = targets.reshape(batch_size, -1)

    # split inputs and targets into sub_arrays, split at the rows
    input_batch = np.split(input_batch, num_batches, 1)
    target_batch = np.split(target_batch, num_batches, 1)

    # stack arrays and reshape
    batches = np.stack(list(zip(input_batch, target_batch)))
    batches = batches.reshape(num_batches, 2, batch_size, seq_length)

    # return numpy array of batched int_text
    return batches


def train_model():
    batches = get_batches(int_text, batch_size, seq_length)

    with tf.Session(graph=train_graph) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch_i in range(num_epochs):
            state = sess.run(initial_state, {input_text: batches[0][0]})

            for batch_i, (x, y) in enumerate(batches):
                feed = {
                    input_text: x,
                    targets: y,
                    initial_state: state,
                    lr: learning_rate}
                train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

                # Show every <show_every_n_batches> batches
                if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                    print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                        epoch_i,
                        batch_i,
                        len(batches),
                        train_loss))

        # Save Model
        saver = tf.train.Saver()
        saver.save(sess, save_dir)
        print('Model Trained and Saved')

    # Save parameters for checkpoint
    helper.save_params((seq_length, save_dir))


def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """

    # get tensors using their names
    input_tensor = loaded_graph.get_tensor_by_name('input:0')
    initial_state = loaded_graph.get_tensor_by_name('initial_state:0')
    final_state = loaded_graph.get_tensor_by_name('final_state:0')
    probability = loaded_graph.get_tensor_by_name('probs:0')

    # return tuple of tensors
    return input_tensor, initial_state, final_state, probability


def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """

    # find index of largest probability
    index = max(enumerate(probabilities), key=operator.itemgetter(1))[0]

    # return predicted word using index
    return int_to_vocab[index]


def generate_script(gen_length=200, prime_word='moe_szyslak'):
    """
    Generate the script of any length and with prime speaker
    :param gen_length: number of words to be used in script
    :param prime_word: homer_simpson, moe_szyslak, or Barney_Gumble
    :print: generated tv script
    """

    loaded_graph = tf.Graph()

    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_dir + '.meta')
        loader.restore(sess, load_dir)

        # Get Tensors from loaded model
        input_text, initial_state, final_state, probs = get_tensors(loaded_graph)

        # Sentences generation setup
        gen_sentences = [prime_word + ':']
        prev_state = sess.run(initial_state, {input_text: np.array([[1]])})

        # Generate sentences
        for n in range(gen_length):
            # Dynamic Input
            dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
            dyn_seq_length = len(dyn_input[0])

            # Get Prediction
            probabilities, prev_state = sess.run(
                [probs, final_state],
                {input_text: dyn_input, initial_state: prev_state})

            pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

            gen_sentences.append(pred_word)

        # Remove tokens
        tv_script = ' '.join(gen_sentences)
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            tv_script = tv_script.replace(' ' + token.lower(), key)
        tv_script = tv_script.replace('\n ', '\n')
        tv_script = tv_script.replace('( ', '(')

        print(tv_script)


if __name__ == '__main__':

    helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
    int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()

    # Hyperparameters
    # Number of Epochs
    num_epochs = 200
    # Batch Size
    batch_size = 128
    # RNN Size
    rnn_size = 400
    # Embedding Dimension Size
    embed_dim = 400
    # Sequence Length
    seq_length = 16
    # Learning Rate
    learning_rate = 0.005
    # Show stats for every n number of batches
    show_every_n_batches = 50

    # build graph
    train_graph = tf.Graph()
    with train_graph.as_default():
        vocab_size = len(int_to_vocab)
        input_text, targets, lr = get_inputs()
        input_data_shape = tf.shape(input_text)
        cell, initial_state = get_init_cell(input_data_shape[0], rnn_size)
        logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

        # Probabilities for generating words
        probs = tf.nn.softmax(logits, name='probs')

        # Loss function
        cost = seq2seq.sequence_loss(
            logits,
            targets,
            tf.ones([input_data_shape[0], input_data_shape[1]]))

        # Optimizer
        optimizer = tf.train.AdamOptimizer(lr)

        # Gradient Clipping
        gradients = optimizer.compute_gradients(cost)
        capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
        train_op = optimizer.apply_gradients(capped_gradients)

        # train the model
        train_model()

        # save the trained model and parameters
        _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
        seq_length, load_dir = helper.load_params()

        # generate a script using the trained graph
        generate_script()
