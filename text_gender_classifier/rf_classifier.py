import pickle
from .text_processing import extract_features
import numpy as np
import pandas as pd

feature_cols = [
    'mean_usage_[URL]',
    'mean_usage_[ADDRESS]',
    'mean_usage_[NUMBER]', 'mean_usage_[QUOTE]',
    'mean_usage_[PUNCEM]', 'mean_usage_[REMOVED]', 'mean_usage_,',
    'mean_usage_!', 'mean_usage_?', 'mean_usage_:',
    'mean_usage_-', 'mean_usage_(', 'mean_usage_)', 'mean_usage_ADJ',
    'mean_usage_ADP', 'mean_usage_ADV',
    'mean_usage_NOUN', 'mean_usage_DET',
    'mean_usage_NUM', 'mean_usage_VERB',
    'mean_usage_PART', 'mean_usage_PRON', 'mean_usage_SCONJ',
    'mean_usage_sentence_length', 'mean_token_length',
    'sentences_count',
]

def __prepare_X(input_dict, required_features):
    array = np.array([input_dict[feat] for feat in required_features])
    X = pd.DataFrame(array.reshape(1, -1), columns=required_features)
    return X

def load_classifier(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    
    def predict(text):
        features = extract_features(text)
        X = __prepare_X(features, feature_cols)
        return model.predict(X)

    return predict
