import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sys
import os
import plotly.graph_objects as go

IS_OFFLINE = True

if IS_OFFLINE:
    sys.path.append(os.getcwd())
import src.data.preprocess as dp

MODEL_PATH = {
    'MODEL_FOLDER' : 'models',
    'MODEL_NAME' : 'model_rf.pickle',
    'FEATURE_NAME' : 'features_names.pickle',
    'POS_VECTORIZER_NAME' : 'pos_vectorizer.pickle',
    'TEXT_VECTORIZE_NAME' : 'text_vectorizer.pickle'
}

st.title('Foot ⚽ Bait Blocker')

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


# load the model
model, features_list, pos_vectorizer, text_vectorizer = load_model(MODEL_PATH)


left_columns, right_columns = st.columns(2)

with left_columns:
    st.write("Entrez le titre de l'article \n")
    headline = st.text_area("Titre de l'article", value='Le PSG est dos au mur')

with right_columns:
    df = pd.DataFrame([headline], columns=['headline'])
    x = dp.data_handling(df, pos_vectorizer, text_vectorizer, features_list)
    y_pred = model.predict(x)
    y_pred_proba = model.predict_proba(x)[:, 1]
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=y_pred_proba.tolist()[0]*100,
        title={'text' : 'Probabilité de clickbait', 'font_size' : 18},
        domain={'x' : [0, 1], 'y' : [0,1]},
        number={'font_size': 24, 'suffix' : '%'},
        gauge={
            'axis' : {'range' : [0,100]},
            'bar': {'color': "darkorange" if y_pred_proba.tolist()[0] > 0.5 else 'darkcyan'}
        }
    ))
    fig.update_layout(
        autosize=False,
        width=230,
        height=230,
        margin=dict(
            l=0,
            r=0,
            b=0,
            t=0,
            pad=0
        ),
    )
    st.plotly_chart(fig)
    #st.write(y_pred_proba.tolist())

if y_pred_proba:
    if y_pred_proba.tolist()[0] < 0.5:
        st.markdown('<h1 style="font-size: 70px; text-align:center">😊<p style="font-size: 28px; color: darkcyan">Pas de clickbait</p></h1>', unsafe_allow_html=True)
    else:
        st.markdown('<h1 style="font-size: 70px; text-align:center">🤬<p style="font-size: 28px; color: darkorange">Clickbait</p></h1>', unsafe_allow_html=True)