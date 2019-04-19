from datetime import datetime
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, precision_score
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import model_from_json
# internal lib
from sentiment.config import HOME_PATH
from sentiment.ai.models.cnn import TextCNN
from sentiment.ai.myutil import tokenize, make_embedding, text_to_sequences


OPTIMAL_THRESHOLD = 0.5


class DefaultSetting():
    train_data = pd.read_csv(HOME_PATH/'data/train.csv')
    embedding_path = HOME_PATH/'embeddings/baomoi.model.bin'
    comment_texts = tokenize(train_data['comment'])
    max_features = 40000
    embed_size, word_map, embedding_mat = make_embedding(comment_texts, embedding_path, max_features)


class SimpleModel():
    def load_model(self, model_path):
        with open(model_path/'model_architecture.json', 'r') as f:
            model = model_from_json(f.read())
        model.load_weights(model_path/'model_weights.hdf5')
        with open(model_path/"wordmap.json") as pickle_in:
            word_map = json.load(pickle_in)
        self.model = model
        self.word_map = word_map

    def create_model(self, train_path, embedding_path, max_features=40000, 
                filter_sizes={2,3,4,5}, batch_size=16, trainable=True, use_additive_emb=False):
        train_data = pd.read_csv(train_path)
        embed_size = DefaultSetting.embed_size
        self.word_map = word_map = DefaultSetting.word_map
        embedding_mat = DefaultSetting.embedding_mat

        # make balance negative + positive
        train_neg = train_data[train_data.label == 1]
        train_pos = train_data[train_data.label == 0]
        neg_count = train_neg.shape[0]
        pos_count = train_pos.shape[0]
        if pos_count > neg_count:
            replicate_data = train_neg.sample(pos_count - neg_count)
        else:
            replicate_data = train_pos.sample(neg_count - pos_count)
        train_data = pd.concat([train_pos, train_neg, replicate_data])
        train_tokenized_texts = tokenize(train_data['comment'])
        labels = train_data['label'].values.astype(np.float16).reshape(-1, 1)
        train_tokenized_texts, val_tokenized_texts, labels_train, labels_val = train_test_split(train_tokenized_texts, labels, test_size = 0.05)
        texts_id_train = text_to_sequences(train_tokenized_texts, word_map)
        texts_id_val = text_to_sequences(val_tokenized_texts, word_map)
        # save model
        model_name = datetime.now().strftime("%Y%m%d-%H%M")
        model_path = HOME_PATH/'output/models/'/model_name
        model_path.mkdir(parents=True, exist_ok=True)
        checkpoint = ModelCheckpoint(filepath=str(model_path/'model_weights.hdf5'),
            monitor='val_f1', verbose=1, mode='max', save_best_only=True
        )
        early = EarlyStopping(monitor='val_f1', mode='max', patience=5)
        callbacks_list = [checkpoint, early]
        epochs = 100

        model = TextCNN(
            embeddingMatrix=embedding_mat,
            embed_size=embed_size,
            max_features=embedding_mat.shape[0],
            filter_sizes=filter_sizes,
            trainable=trainable,
            use_additive_emb=use_additive_emb
        )
        model.fit(
            texts_id_train, labels_train,
            validation_data=(texts_id_val, labels_val),
            callbacks=callbacks_list,
            epochs=epochs,
            batch_size=batch_size
        )
        model.load_weights('{}/model_weights.hdf5'.format(model_path))
        self.model = model
        prediction_prob = model.predict(texts_id_val)
        prediction = (prediction_prob > OPTIMAL_THRESHOLD).astype(np.int8)
        print( 'F1 validation score: {}'.format(f1_score(prediction, labels_val)) )

        # Save the model architecture
        with open('{}/model_architecture.json'.format(model_path), 'w') as f:
            f.write(model.to_json())
        # save word_map
        with open("{}/wordmap.json".format(model_path),"w") as pickle_out:
            json.dump(self.word_map, pickle_out)

    def predict(self, input, output):
        data = pd.read_csv(input)
        data['comment'] = data['comment'].astype(str).fillna(' ')
        _texts_id = text_to_sequences(tokenize(data['comment']), self.word_map)
        _prob = self.model.predict(_texts_id)
        _pred = (_prob > OPTIMAL_THRESHOLD).astype(np.int8)
        data['prob'] = _prob
        data['pred'] = _pred
        data.to_csv(output, index=False, encoding='utf-8')

    def predict_text(self, text):
        text_np = np.array([text])
        _texts_id = text_to_sequences(tokenize(text_np), self.word_map)
        _prob = self.model.predict(_texts_id)
        return _prob[0,0]

if __name__ == '__main__':
    from pathlib import Path

    # path = Path("/Users/vanvt/2019/learn/simple-sentiment")
    path = Path("/content/simple-sentiment/")
    data_path = path/'data/train.csv'
    embedding_path = path/'embeddings/baomoi.model.bin'
    model = SimpleModel()
    model.create_model(data_path, embedding_path)
    print('FINISH')