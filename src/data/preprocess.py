# librairies
import re
from  typing import List
import pandas as pd
import cleantext as ct
import spacy
from nltk.stem.snowball import  SnowballStemmer

# tools
fr = spacy.load('fr_core_news_sm')
stopwords = fr.Defaults.stop_words
stemmer = SnowballStemmer(language='french')

# cleaning function
def clean_headline(text) -> str:

     # convert to string
     text = str(text)

     # remove emoji
     text = ct.clean(text, no_emoji=True)

     # remove hashtag
     text = re.sub(r'#.*\s', '', text)

     # remove link
     text = re.sub(r'(https://[\.|/|a-zA-Z|0-9]+|[a-zA-Z|0-9]+\.[a-zA-Z|0-9]+/[a-zA-Z|0-9]+)', ' ', text)

     # remove useless space
     text = text.strip()

     return text

# Preprocess text
def preprocess_text(text : str) -> List:

     # lower the text
     text = text.lower()

     # lemmatization
     doc = fr(text)
     tokens = [ token.lemma_ for token in doc ]
     text = ' '.join(tokens)

     # remove punctuation
     text = re.sub(r'[^\w\s]', ' ', text)

     # remove digits
     text = re.sub(r'\d', '', text).strip()

     # remove stop words
     ls = [ word for word in text.split() if word not in stopwords ]

     # remove single character words
     ls = [word for word in ls if len(word)>= 2]

     # stemming
     tokens = [ stemmer.stem(token) for token in ls ]

     return tokens


# preprocess POS
def preprocess_text_pos(text : str) -> List:

     # lower the headline
     text = text.lower()

     # pos tagging
     doc = fr(text)
     tokens = [ token.pos_ if token.ent_type_ =='' else token.ent_type_ for token in doc ]

     return tokens

# preprocess pipeline
def preprocess(data):
     data['headline_cleaned'] = data['headline'].apply(clean_headline)
     data['headline_tokenized'] = data['headline_cleaned'].apply(preprocess_text)
     data['headline_pos_tokenized'] = data['headline_cleaned'].apply(preprocess_text_pos)
     return data

# feature engineering
def features_eng(data : pd.DataFrame, pos_vectorizer, text_vectorizer, features_keep) -> pd.DataFrame:

     # headline length
     data['headline_length'] = data['headline_cleaned'].apply(lambda x : len(str(x)))

     # is a question
     data['is_question'] = data['headline_cleaned'].apply(lambda x : '?' in x)
     data['is_question'] = data['is_question'].astype(int)

     # is an exclamation
     data['is_exclamation'] = data['headline_cleaned'].apply(lambda x : '!' in x)
     data['is_exclamation'] = data['is_exclamation'].astype(int)

     # have a score
     data['have_score'] = data['headline_cleaned'].apply(
          lambda x : re.search(r'\d(\s*)-(\s*)\d', x) is not None)
     data['have_score'] = data['have_score'].astype(int)

     # POS td-idf
     corpus = data['headline_pos_tokenized'].apply(lambda x : ' '.join(x)).values.tolist()
     X = pos_vectorizer.transform(corpus)
     post_matrix_tfidf = pd.DataFrame.sparse.from_spmatrix(X,
                              columns=pos_vectorizer.get_feature_names_out())

     # Text tf-idf
     corpus = data['headline_tokenized'].apply(lambda x : ' '.join(x)).values.tolist()
     X_text = text_vectorizer.transform(corpus)
     text_matrix_tfidf = pd.DataFrame.sparse.from_spmatrix(X_text,
                              columns=text_vectorizer.get_feature_names_out())

     data_final = pd.concat([data, post_matrix_tfidf,
                             text_matrix_tfidf[features_keep]], axis=1)

     return data_final


### data pipelie
def data_handling(data, pos_vectorizer, text_vectorizer, features):
     data = preprocess(data)
     data = features_eng(data, pos_vectorizer, text_vectorizer, features)

     x = data.drop(columns=['headline', 'headline_cleaned', 'headline_tokenized',
                            'headline_pos_tokenized'])

     return x