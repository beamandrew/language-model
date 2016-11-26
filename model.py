def sampled_loss(y_true, y_pred):
    labels = tf.reshape(y_true, [-1, 1])
    # We need to compute the sampled_softmax_loss using 32bit floats to
    # avoid numerical instabilities.
    local_w_t = tf.cast(w_t, tf.float32)
    local_b = tf.cast(b, tf.float32)
    local_inputs = tf.cast(y_pred, tf.float32)
    return tf.cast(
        tf.nn.sampled_softmax_loss(local_w_t, local_b, local_inputs, labels,
                                   25, vocab_size),
        np.float32)


import tensorflow as tf
from keras.models import Model
from keras.layers import Dense, Dropout, Embedding, LSTM, Input, TimeDistributed, Activation
from keras.optimizers import Adam,SGD

def get_model(params):
    input = Input(shape=(seq_len,), dtype='int32', name='main_input')
    f = Embedding(vocab_size+1, 128, input_length=seq_len)(input)
    f = LSTM(output_dim=512,return_sequences=True,name='rnn_1')(f)
    f = LSTM(output_dim=512, return_sequences=True, name='rnn_2')(f)
    f = TimeDistributed(Dense(params['num_classes'],activation='softmax'))(f)
    model = Model(input,f)
    model.compile(optimizer='rmsprop',loss='categorical_crossentropy')
    return model