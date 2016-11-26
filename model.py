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

def get_model_tf(params):
    batch_size = params['batch_size']
    seq_len = params['seq_len']
    vocab_size = params['vocab_size']
    input = tf.placeholder(tf.float32,shape=(batch_size,seq_len))
    rnn = Embedding(vocab_size+1, 128, input_length=seq_len)(input)
    rnn = LSTM(output_dim=512,return_sequences=True,name='rnn_1')(rnn)
    rnn_output = tf.unpack(rnn,axis=1)
    w_proj = tf.Variable(tf.zeros([20000,512]))
    b_proj = tf.Variable(tf.zeros([20000]))
    labels = tf.placeholder(tf.int64,shape=(batch_size,seq_len))
    losses = []
    outputs = []
    for t in range(seq_len):
        rnn_t = rnn_output[t]
        y_t = tf.reshape(labels[:,t],(batch_size,1))
        step_loss = tf.nn.sampled_softmax_loss(weights=w_proj, biases=b_proj, inputs=rnn_t,
                                                    labels=y_t, num_sampled=25, num_classes=params['vocab_size'])
        losses.append(step_loss)
        outputs.append(tf.matmul(rnn_t,tf.transpose(w_proj)) + b_proj)
    return losses,outputs