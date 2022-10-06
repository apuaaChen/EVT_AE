import numpy as np
import xgboost as xgb

def collect_dataset(features, labels, existing_feature = None, existing_label = None, num_feature = 0):
    if existing_feature is None:
        existing_feature = np.zeros((0,num_feature))
    else:
        assert len(existing_feature.shape) == 2, "Error in cost model: existing_feature shape is wrong"
        assert existing_feature.shape[1] == num_feature, "Error in cost model: existing_feature dimension does not match num_feature"
    features = np.vstack([existing_feature, features])
    labels = np.array(list(existing_label) + list(labels))
    return (features, labels)

def train_model(features, labels):
    model = xgb.XGBRegressor()
    model.fit(features, labels)
    return model

def predict(model, parameters):
    return model.predict(parameters)

if __name__ == "__main__":
    features = np.random.normal(size=(2,11))
    labels = []
    for i in range(2):
        tmp = features[i].sum()
        labels.append(tmp)
    labels = np.array(labels)
    model = train_model(features, labels)
    for i in range(20):
        new_features = np.random.normal(size=(1,11))
        print("expected: ", new_features[0].sum())
        prediction = predict(model, new_features)
        print("prediction: ", prediction[0])