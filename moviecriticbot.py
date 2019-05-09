# -*- coding: utf-8 -*-
from __future__ import absolute_import
from flask import Flask, render_template, flash, request, Markup
#from wtforms import Form, TextField, TextAreaField, validators, StringField, SubmitField
from wtforms import TextField,validators,Form
from textgenrnn import textgenrnn
from keras import backend as K

#Load model

def generate_phrases(sent,lang):
    #Before prediction
    K.clear_session()
    textgen = textgenrnn(weights_path='models/{}_{}/top_{}_IMDb_sentences_weights_{}.hdf5'.format(sent,lang,sent,lang),
                         vocab_path='models/{}_{}/top_{}_IMDb_sentences_vocab_{}.json'.format(sent,lang,sent,lang),
                         config_path='models/{}_{}/top_{}_IMDb_sentences_config_{}.json'.format(sent,lang,sent,lang))
   
    #Generate text   
    # this temperature schedule cycles between 1 very unexpected token, 1 unexpected token, 2 expected tokens, repeat.
    # changing the temperature schedule can result in wildly different output!    
    temperature = [0.4,0.2]
    prefix = ''   # if you want each generated text to start with a given seed text
    x = textgen.generate(temperature=temperature,n=1,
                         max_gen_length=150,prefix=prefix,return_as_list=True)
    #After prediction
    K.clear_session()
    return x

# App config.
DEBUG = True
app = Flask(__name__)

app.config.from_object(__name__)
app.config['SECRET_KEY'] = '7d441f27d441f27567d441f2b6176a'


def get_snippet(lang,sent):
    if sent == 'neg':
        bot = 'AngryBotFlat-Sm_s.png'
    if sent == 'pos':
        bot = 'Robot-60-TopHat-A_s.png'
    phrase = generate_phrases(sent,lang)[0]
    snippet = """<img src="static\{}"></img><div class="phrase_{}">{}</div>""".format(bot,sent,phrase)
    return snippet


@app.route("/", methods=['GET', 'POST'])

def contact():
    form = Form
    if request.method == 'POST':
        if request.form.get('English'):
            if request.form['submit_button'] == 'Negative':
                snippet = get_snippet('en','neg')
                flash(Markup(snippet))
                pass 
            elif request.form['submit_button'] == 'Positive':
                snippet = get_snippet('en','pos')            
                flash(Markup(snippet))
                pass 

        if request.form.get('Spanish'):
            if request.form['submit_button'] == 'Negative':
                snippet = get_snippet('es','neg')
                flash(Markup(snippet))
                pass 
            elif request.form['submit_button'] == 'Positive':
                snippet = get_snippet('es','pos')
                flash(Markup(snippet))
                pass 
        
        if request.form.get(None):
            flash('Please select a language. Movie critic bot is bilingual')
            pass # unknown

    return render_template('home.html',form=form)

if __name__ == "__main__":
    app.run(debug=True)
