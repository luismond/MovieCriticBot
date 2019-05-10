# -*- coding: utf-8 -*-
from __future__ import absolute_import
from flask import Flask, render_template, flash, request, Markup
from wtforms import TextField,validators,Form
from textgenrnn import textgenrnn
from keras import backend as K

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(lowercase=True)
from joblib import load


# =============================================================================
# PREDICT
# =============================================================================

# English
vocab_en = open('predict/en/Sentiment_Classifier_en_neural_MLP_vocab.txt','r',encoding='utf8').read().split('\n')
vectorizer_en = TfidfVectorizer(lowercase=True)
vectorizer_en.fit(vocab_en)
clf_en = load('predict/en/Sentiment_Classifier_en_neural_MLP.joblib') 

# Spanish
vocab_es = open('predict/es/Sentiment_Classifier_es_neural_MLP_vocab.txt','r',encoding='utf8').read().split('\n')
vectorizer_es = TfidfVectorizer(lowercase=True)
vectorizer_es.fit(vocab_es)
clf_es = load('predict/es/Sentiment_Classifier_es_neural_MLP.joblib') 


# =============================================================================
# GENERATE
# =============================================================================

#Load models
def generate_phrases(sent,lang):
    #Before prediction
    K.clear_session()
    textgen = textgenrnn(weights_path='models/{}_{}/top_{}_IMDb_sentences_weights_{}.hdf5'.format(sent,lang,sent,lang),
                         vocab_path='models/{}_{}/top_{}_IMDb_sentences_vocab_{}.json'.format(sent,lang,sent,lang),
                         config_path='models/{}_{}/top_{}_IMDb_sentences_config_{}.json'.format(sent,lang,sent,lang))
    #Generate text   
    temperature = [0.4,0.2]
    prefix = ''
    phrase = textgen.generate(temperature=temperature,n=1,
                         max_gen_length=90,prefix=prefix,return_as_list=True)
    #After prediction
    K.clear_session()
    return phrase

# App config.
DEBUG = False
app = Flask(__name__)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'

def get_snippet(lang,sent):
    if sent == 'neg':
        bot = 'AngryBotFlat-Sm_s.png'
    if sent == 'pos':
        bot = 'Robot-60-TopHat-A_s.png'
    phrase = generate_phrases(sent,lang)[0]
    snippet = """<div class="phrase_{}">{}</div><img src="static\{}"></img>""".format(sent,phrase,bot)
    return snippet


def get_snippet_pred(sent,name,lang):
    if sent == 'neg':
        bot = 'AngryBotFlat-Sm_s.png'
        if lang == 'en':
            phrase = 'This review is negative!'
        if lang == 'es':
            phrase = '¡Esta opinión es negativa!'
    if sent == 'pos':
        bot = 'Robot-60-TopHat-A_s.png'
        if lang == 'en':
            phrase = 'This review is positive!'
        if lang == 'es':
            phrase = '¡Esta opinión es positiva!'
    snippet = """<div class="phrase_{}">{}</div></br><img src="static\{}"></img><div>{}</div>""".format(sent,name,bot,phrase)
    return snippet


@app.route("/", methods=['GET', 'POST'])

def home():
    form = Form
    if request.method == 'POST':

        if request.form['submit_button'] == 'Negative':
            snippet = get_snippet('en','neg')
            flash(Markup(snippet))
            pass 
        elif request.form['submit_button'] == 'Positive':
            snippet = get_snippet('en','pos')            
            flash(Markup(snippet))
            pass 

    return render_template('generate_en.html',form=form)



@app.route("/en_generate", methods=['GET', 'POST'])

def generate_en():
    form = Form
    if request.method == 'POST':

        if request.form['submit_button'] == 'Negative':
            snippet = get_snippet('en','neg')
            flash(Markup(snippet))
            pass 
        elif request.form['submit_button'] == 'Positive':
            snippet = get_snippet('en','pos')            
            flash(Markup(snippet))
            pass 

    return render_template('generate_en.html',form=form)

@app.route("/es_generate", methods=['GET', 'POST'])

def generate_es():
    form = Form
    if request.method == 'POST':

        if request.form['submit_button'] == 'Negativa':
            snippet = get_snippet('es','neg')
            flash(Markup(snippet))
            pass 
        elif request.form['submit_button'] == 'Positiva':
            snippet = get_snippet('es','pos')            
            flash(Markup(snippet))
            pass 

    return render_template('generate_es.html',form=form)


class ReusableForm(Form):
    name = TextField('word:', validators=[validators.required()])

@app.route("/en_predict", methods=['GET', 'POST'])

def predict_en():
    form = ReusableForm(request.form)
    if request.method == 'POST':
        name=request.form['name']
        if form.validate():
            if len(name)>30:
                pred = clf_en.predict(vectorizer_en.transform([name]))
                if pred == 0:
                    snippet = get_snippet_pred('neg',name,'en')
                    flash(Markup(snippet))
                if pred == 1:
                    snippet = get_snippet_pred('pos',name,'en')
                    flash(Markup(snippet))
            else:
                flash('Please enter at least 30 characters')

    return render_template('predict_en.html',form=form)

@app.route("/es_predict", methods=['GET', 'POST'])

def predict_es():
    form = ReusableForm(request.form)
    if request.method == 'POST':
        name=request.form['name']
        if form.validate():
            if len(name)>30:
                pred = clf_es.predict(vectorizer_es.transform([name]))
                if pred == 0:
                    snippet = get_snippet_pred('neg',name,'es')
                    flash(Markup(snippet))
                if pred == 1:
                    snippet = get_snippet_pred('pos',name,'es')
                    flash(Markup(snippet))
            else:
                flash('Por favor, escribe por lo menos 30 caracteres')

    return render_template('predict_es.html',form=form)

if __name__ == "__main__":
    app.run(debug=True)
