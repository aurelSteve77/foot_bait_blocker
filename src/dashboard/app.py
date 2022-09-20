import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
sys.path.append('..')
import src.data.preprocess as dp

st.write(os.getcwd())

MODEL_PATH = {
    'MODEL_FOLDER' : '..\..\models',
    'MODEL_NAME' : 'model_rf.pickle',
    'FEATURE_NAME' : 'features_names.pickle',
    'POS_VECTORIZER_NAME' : 'pos_vectorizer.pickle',
    'TEXT_VECTORIZE_NAME' : 'text_vectorizer.pickle'
}

st.title('Foot Bait Blocker')

@st.cache(allow_output_mutation=True)
def load_model(model_path):

    # model
    model_fs = open(f"{model_path['MODEL_FOLDER']}/{model_path['MODEL_NAME']}", 'rb')
    model = pickle.load(model_fs)
    model_fs.close()

    # features list
    features_fs = open(f"{model_path['MODEL_FOLDER']}/{model_path['FEATURE_NAME']}", 'rb')
    features_list = pickle.load(features_fs)
    features_fs.close()

    # POS vectorizer
    pos_vec_fs = open(f"{model_path['MODEL_FOLDER']}/{model_path['POS_VECTORIZER_NAME']}", 'rb')
    pos_vectorizer = pickle.load(pos_vec_fs)
    pos_vec_fs.close()

    # Text vectorizer
    text_vectorizer_fs = open(f"{model_path['MODEL_FOLDER']}/{model_path['TEXT_VECTORIZE_NAME']}", 'rb')
    text_vectorizer = pickle.load(text_vectorizer_fs)
    text_vectorizer_fs.close()

    return model, features_list, pos_vectorizer, text_vectorizer

# loading model text
model_load_state = st.text('Loading the model ....')

# load the model
model, features_list, pos_vectorizer, text_vectorizer = load_model(MODEL_PATH)

# notify that model sucessful loader
model_load_state.text('Loading the model .... done!')

left_columns, right_columns = st.columns(2)

with left_columns:
   headline = st.text_area("Titre de l'article", value='Le PSG est dos au mur')

with right_columns:
    st.write('Right text')
    df = pd.DataFrame([headline], columns=['headline'])
    x = dp.data_handling(df, pos_vectorizer, text_vectorizer, features_list)
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:, 1]
    st.write(y_pred_proba.tolist())