import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import plotly.graph_objects as go
#from sklearn.externals import joblib #deprecated, using joblib directly instead
import joblib
from sqlalchemy import create_engine


app = Flask(__name__)

def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
engine = create_engine('sqlite:///data/DisasterResponse.db')
df = pd.read_sql_table('categories', engine)

# load model
model = joblib.load("models/xgboost.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    df_categories = df.drop(columns=['message', 'original', 'genre', 'id']).sum().sort_values(ascending=False)
    df_categories.name = 'Total'
    categories_counts = df_categories.values
    categories_names = df_categories.index

    df_genre_categories = df.drop(columns=['id']).groupby('genre').sum()
    df_genre_categories = df_genre_categories.append(df_categories)
    df_genre_categories.sort_values('Total', axis=1, ascending=False, inplace=True)
    df_direct = df_genre_categories.loc['direct'].values
    df_news = df_genre_categories.loc['news'].values
    df_social = df_genre_categories.loc['social'].values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=df_news,
                    marker=dict(color='rgb(85,201,159)'),
                    name='News'
                ),
                Bar(
                    x=categories_names,
                    y=df_direct,
                    marker=dict(color='rgb(161, 204, 230)'),
                    name='Direct'
                ),
                Bar(
                    x=categories_names,
                    y=df_social,
                    marker=dict(color='rgb(245, 184, 162)'),
                    name='Social'
                )
            ],

            'layout': {
                'title': 'Message Categories per Genre',
                'yaxis': {
                    'title': "Count",
                    'showgrid' : False                
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle' : -45,
                    'automargin' : True                       
                },
                'barmode' : 'stack'
                
                
            }
        },
        {
            'data': [
                Bar(
                    x=categories_names,
                    y=categories_counts,
                    marker=dict(color='rgb(85,201,159)'),
                    hoverinfo = None
                )
            ],

            'layout': {
                'title': 'Distribution of Message Categories',
                'yaxis': {
                    'title': "Count",
                    'showgrid' : False                
                },
                'xaxis': {
                    'title': "Categories",
                    'tickangle' : -45,
                    'automargin' : True
                                        
                }
                
                
            }
        },
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


# def main():
#     app.run(host='0.0.0.0', port=3001, debug=True)


# if __name__ == '__main__':
#     main()