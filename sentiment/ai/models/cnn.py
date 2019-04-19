from keras.models import Model
import keras.backend as K
from keras.layers import Layer, \
    Dense, Embedding, Input, \
    Conv1D, MaxPool1D, \
    Dropout, BatchNormalization, \
    Bidirectional, CuDNNLSTM, \
    Concatenate, Flatten, Add

class AdditiveLayer(Layer):
    def __init__(self):
        super(AdditiveLayer, self).__init__()

    def build(self, input_shape):
        self._w = self.add_weight(
            name = "w",
            shape = (1, input_shape[-1]),
            initializer="constant",
            trainable=True
        )
        super(AdditiveLayer, self).build(input_shape)

    def call(self, input):
        return input + self._w

    def compute_output_shape(self, input_shape):
        return input_shape

def f1(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

def TextCNN(embeddingMatrix = None, embed_size = 400, max_features = 20000, maxlen = 100, 
    filter_sizes = {2, 3, 4, 5}, use_fasttext = False, trainable = True, use_additive_emb = False):
    if use_fasttext:
        inp = Input(shape=(maxlen, embed_size))
        x = inp
    else:
        inp = Input(shape = (maxlen, ))
        x = Embedding(input_dim = max_features, output_dim = embed_size, weights = [embeddingMatrix], trainable = trainable)(inp)

    if use_additive_emb:
        x = AdditiveLayer()(x)
        x = Dropout(0.5)(x)


    conv_ops = []
    for filter_size in filter_sizes:
        conv = Conv1D(128, filter_size, activation = 'relu')(x)
        pool = MaxPool1D(5)(conv)
        conv_ops.append(pool)

    concat = Concatenate(axis = 1)(conv_ops)
    # concat = Dropout(0.1)(concat)
    concat = BatchNormalization()(concat)


    conv_2 = Conv1D(128, 5, activation = 'relu')(concat)
    conv_2 = MaxPool1D(5)(conv_2)
    conv_2 = BatchNormalization()(conv_2)
    # conv_2 = Dropout(0.1)(conv_2)

    conv_3 = Conv1D(128, 5, activation = 'relu')(conv_2)
    conv_3 = MaxPool1D(5)(conv_3)
    conv_3 = BatchNormalization()(conv_3)
    # conv_3 = Dropout(0.1)(conv_3)


    flat = Flatten()(conv_3)

    op = Dense(64, activation = "relu")(flat)
    # op = Dropout(0.5)(op)
    op = BatchNormalization()(op)
    op = Dense(1, activation = "sigmoid")(op)

    model = Model(inputs = inp, outputs = op)
    model.compile(loss = 'binary_crossentropy', optimizer = 'adam', metrics = ['accuracy', f1])
    print(model.summary())
    return model
