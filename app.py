import sys
sys.path.append("/Users/lesleymi/data_science_portfolio/IMDB_Sentiment_Analysis/src")
# custom functions
#import imdb_functions as imdb

# data wrangling
import pandas as pd
import numpy as np

# nlp
from nltk.probability import FreqDist

# viz
import plotly.express as px
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# Plotly Dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output
import base64

# functions
def make_top_n(df, n, polarity):
    plot = px.bar(df,
                  x='word_count',
                  y='words',
                  labels={'words': 'Top ' + str(n) + ' Words', 'word_count': 'Word Count'},
                  title='Top ' + str(n) + ' Most Frequent Words in ' + polarity + ' Reviews')
    return plot

# load data
train = pd.read_parquet("data/Train.parquet")
# replace 0/1s with human readable labels
train = train.replace({'label':{0:'negative',
                                1: 'positive'}})
# replace 0/1s with human readable labels
train_clean = pd.read_parquet("data/train_clean.parquet")
train_clean = train_clean.replace({'label':{0:'negative',
                                1: 'positive'}})


# get a negative review
neg_rev = train.query('label == "negative"').text.values

# get a positive review
pos_rev = train.query('label == "positive"').text.values

# combine the documents into a single bag of words
clean_docs_bow = np.concatenate(train_clean.tokenized_docs)

# filter for pos/neg reviews
train_clean_pos = train_clean.query('label == "positive"').tokenized_docs.to_list()
train_clean_neg = train_clean.query('label == "negative"').tokenized_docs.to_list()

# combine the documents into a single bag of words
positive_bow = np.concatenate(train_clean_pos)
negative_bow = np.concatenate(train_clean_neg)


# how many positive/negative reviews are there?
df = (train.groupby(by='label')
          .count()
          .reset_index()
          .rename(columns={'text':'count'}))

# plot he count of pos/neg labels
plot1 = px.bar(df,
       x='label',
       y='count',
       color='label',
       title='Count of Positive and Negative Reviews',
       labels={'label': 'Movie Review Sentiment',
               'count': 'Count of Reviews'})
plot1.show('png')

# visualize document length by sentiment label
plot4 = px.histogram(train_clean,
                     x='num_tokens',
                     color='label',
                     labels={'num_tokens':'Number of Tokens'},
                     title='Distribution of Cleaned Movie Review Length')
plot4.show('png')




## intro text

markdown_text1 = """
This app displays visualizations of the IMDB movie review dataset found [here on Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format).
All the code for this analysis can be found on GitHub [here](https://github.com/aromatic-toast/IMDB_Sentiment_Analysis). The data displayed here is for 40,000 movie reviews
labeled as either positive or negative reviews. 
"""

markdown_text2 = """
Use the tabs to navigate to an example of a positive or negative movie review. 
Click the arrows inside the box up and down to browse through different reviews or enter a number
directly between 0 and 19,000. 
"""
# add positive and negative review examples to a table
table_header = [html.Thead(
    html.Tr([html.Th("Positive Review"), html.Th('Negative Review')])
)]
row1 = html.Tr([html.Td(pos_rev), html.Td(neg_rev)])
table_body = [html.Tbody([row1])]
table = dbc.Table(table_header + table_body,
                  bordered=True,
                  striped=True)

# get the vocabulary and their frequencies
all_words = (pd.DataFrame(clean_docs_bow, columns=['word_count'])
             .word_count
             .value_counts()
             .reset_index()
             .rename(columns={'index': 'words'}))

pos_words = (pd.DataFrame(positive_bow, columns=['word_count'])
             .word_count
             .value_counts()
             .reset_index()
             .rename(columns={'index': 'words'}))

neg_words = (pd.DataFrame(negative_bow, columns=['word_count'])
             .word_count
             .value_counts()
             .reset_index()
             .rename(columns={'index': 'words'}))

encoded_image = base64.b64encode(open('wordcloud_fig.png', 'rb').read())

PLOTLY_LOGO = "https://images.plot.ly/logo/new-branding/plotly-logomark.png"
navbar = dbc.Navbar(
    [
        html.A(
            # Use row and col to control vertical alignment of logo / brand
            dbc.Row(
                [
                    dbc.Col(html.Img(src=PLOTLY_LOGO, height="40px")),
                    dbc.Col(dbc.NavbarBrand("IMDB Movie Reviews", className="ml-2")),
                ],
                align="center",
                no_gutters=True,
            ),
            href="https://plot.ly",
        )
    ],
    color="dark",
    dark=True,
    id='nav-bar'
)

intro_tooltip = dbc.Tooltip(
    # dcc.Markdown(markdown_text1),
    markdown_text1,
    target='nav-bar',
    placement='bottom'
)

tab1_tooltip = dbc.Tooltip(
    markdown_text2,
    target='tab1',
    placement='bottom'
)

# external_stylesheets = [dbc.themes.CERULEAN]
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div(children=[navbar,
                                # html.H3(children='Exploratory Text Analysis'),
                                intro_tooltip,

                                # 3 main tabs
                                dcc.Tabs([
                                    # tab to text documents
                                    dcc.Tab(id='tab1', label='Read Movie Reviews', children=[
                                        dbc.Row(html.Br()),
                                        tab1_tooltip,
                                        # dbc.Row(dbc.Col([html.H6(markdown_text2)], width=4)),
                                        dbc.Row(html.Br()),
                                        dbc.Row(dbc.Col([
                                            html.Label("Enter an integer below:")
                                        ])),
                                        dbc.Row(dbc.Col([
                                            dcc.Input(id='sample-review',
                                                      value=0,
                                                      placeholder='Input Review Number',
                                                      type='number')], width=4)),
                                        dbc.Row(html.Br()),
                                        dbc.Row(dbc.Col([
                                            dbc.Tabs([
                                                dbc.Tab(id='tab1_content', label='Positive'),
                                                dbc.Tab(id='tab2_content', label='Negative')
                                            ])
                                        ]))
                                    ]),

                                    # tab to hold the summary plots
                                    dcc.Tab(label='Plots', children=[
                                        dbc.Container([
                                            dbc.Row(html.Br()),
                                            dbc.Row([dbc.Col(dcc.Graph(id='plot1', figure=plot1)),
                                                     dbc.Col(dcc.Graph(id='plot4', figure=plot4))]),
                                            dbc.Row(html.Br()),
                                            dbc.Row([
                                                dbc.Col([
                                                    html.H4('Filter Top Words')
                                                ], width=3)
                                            ]),
                                            dbc.Row([
                                                dbc.Col(
                                                    dcc.Dropdown(
                                                        id='drop-down',
                                                        options=[
                                                            {'label': '5', 'value': 5},
                                                            {'label': '10', 'value': 10},
                                                            {'label': '15', 'value': 15}
                                                        ],
                                                        placeholder='Select N', value=10),
                                                    width=1)
                                            ]),
                                            dbc.Row([dbc.Col(dcc.Graph(id='top-overall-words')),
                                                     dbc.Col(dcc.Graph(id='top-pos-words')),
                                                     dbc.Col(dcc.Graph(id='top-neg-words'))]), ],
                                            fluid=True)
                                    ]),

                                    # tab to contain a map
                                    dcc.Tab(label='Wordcloud', children=[
                                        dbc.Row(html.Br()),
                                        # dbc.Row(dbc.Col(html.H5("Some instructions on how to use the map. "))),
                                        dbc.Row(html.Br()),
                                        dbc.Row(dbc.Col([
                                            html.Img(id='wordcloud',
                                                     src='data:image/png;base64,{}'.format(encoded_image.decode()))
                                        ]))

                                    ])
                                ])
                                ])


@app.callback(
    Output('top-overall-words', component_property='figure'),
    Input('drop-down', component_property='value')
)
def get_top_overall_words(n):
    # filter vocabulary to top n words
    filtered_df = all_words.head(n)
    # make the plot
    plot = make_top_n(filtered_df, n, 'All')
    return plot


@app.callback(
    Output('top-pos-words', component_property='figure'),
    Input('drop-down', component_property='value'))
def get_top_pos_words(n):
    # filter vocabulary to top n words
    filtered_df = pos_words.head(n)
    # make the plot
    plot = make_top_n(filtered_df, n, 'Positive')
    return plot


@app.callback(
    Output('top-neg-words', component_property='figure'),
    Input('drop-down', component_property='value'))
def get_top_neg_words(n):
    # filter vocabulary to top n words
    filtered_df = neg_words.head(n)
    # make the plot
    plot = make_top_n(filtered_df, n, 'Negative')
    return plot


@app.callback(
    Output('tab1_content', component_property='children'),
    Input('sample-review', component_property='value'))
def get_pos_review(index):
    tab1_content = dbc.Card(
        dbc.CardBody(
            # filter a document from the positive reviews
            pos_rev[index]
        ))
    return tab1_content


@app.callback(
    Output('tab2_content', component_property='children'),
    Input('sample-review', component_property='value'))
def get_neg_review(index):
    tab2_content = dbc.Card(
        dbc.CardBody(
            # filter a document from the negative reviews
            neg_rev[index]
        ))
    return tab2_content


if __name__ == '__main__':
    app.run_server(debug=True)

