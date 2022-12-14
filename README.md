foot_bait_blocker
==============================

![UI](cover.png)

An app to identify clickbait article in football using machine learning
The language is french

Here is the [demo link](https://aurelsteve77-foot-bait-blocker-srcdashboardapp-hou54y.streamlitapp.com/)


Background
----------
“Kylian Mbappe gifle Pogba”, “Le PSG au bord de l’implosion”, ou encore
“Mbappe trahi par le psg, c’est confirmé” we all have already come across this kind
of article on the internet, given by a large number of websites that are ready to do
anything to get people to click on the article and thus receive an audience. 
This is what we call "putaclic" or "clickbait" for the most informed,
click traps, which consist in putting an enticing title to force users to click to see
the content of the article which often does not correspond to the proposed title to their
great disappointment.


Objectives
----------
The goal of this project is to build a bait blocker on football french news articles,
a kind of "bait'block" using machine learning

The objectives of the projects are the following:
* Build a new dataset of french bait football news using webscrapping
* Build a binary supervised classification model of bait article using headline
* Build a custom web interface allowing to predict if a headline is clickbait or not
* Build a chrome extension to block bait

Steps
-----
| Task               | Technique                                                       | Tools/Packages Used |
|--------------------|-----------------------------------------------------------------|---------------------|
| Data Collection    | Article extraction from websites and twitter using webscrapping | snscrap             |
| Data preprocessing | Clean html, NER, Regex, Lemmatization, Stemming , TFIDF         | Spacy, re, bs4      |
| Data modeling      | Random Forest, LSTM, SVM                                        |                     |
| UI : Web Interface | Streamlit                                                       | Streamlit           |


Data Collection
---------------
Data are collected using web scrapping in two step
#### First step: Get articles headline and link on twitter 
The goal is to have a list of article's headlines and their link on twitter account
categorized as clickbait (foot01, foot mercato...) and other more serious (sofoot, foot365)
This is done using [snscrape](https://github.com/JustAnotherArchivist/snscrape)

#### Second step: Get the body of the article
The goal in this step is to extract the body of each article using the link got
on twitter.
This is done by using beautiful 

Data preprocessing
------------------

Modeling
----------

UI
-----------

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    |   ├── dashboard       <- Web interface app (UI)
    │   │   └── app.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
