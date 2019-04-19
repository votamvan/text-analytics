from sentiment.config import HOME_PATH
from sentiment.train import SimpleModel

# test_data = pd.read_csv(HOME_PATH/'data/test.csv')
# test_tokenizes_texts = tokenize(test_data['comment'].fillna("none").values)

# embed text
embedding_path = HOME_PATH/'embeddings/baomoi.model.bin'
train_path = HOME_PATH/'data/train.csv'
max_features = 40000
params = {
    "train_path": train_path,
    "embedding_path": embedding_path,
    "max_features": max_features,
    "filter_sizes": {2,3,4,5}
}
model1 = SimpleModel()
model1.create_model(**params)

# params["filter_sizes"] = {2,3,4}
# model2 = SimpleModel(**params)

# data_train_path = HOME_PATH/'data/train.csv'
# pred_train_path = HOME_PATH/'output/pred-train-m1.csv'
# model1.predict(data_train_path, pred_train_path)

# data_test_path = HOME_PATH/'data/test.csv'
# pred_test_path = HOME_PATH/'output/pred-test-1.csv'
# model1.predict(data_test_path, pred_test_path)

# pred_train_path = HOME_PATH/'output/pred-train-m2.csv'
# model2.predict(data_train_path, pred_train_path)
# pred_test_path = HOME_PATH/'output/pred-test-2.csv'
# model2.predict(data_test_path, pred_test_path)

