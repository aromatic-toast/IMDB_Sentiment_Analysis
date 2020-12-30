# data wrangling
import pandas as pd
import numpy as np

# viz
import plotly.express as px

# Plotly Dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
import dash_bootstrap_components as dbc
from dash.dependencies import Input, Output, State
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
train = pd.read_parquet("data/Train_subset.parquet")

# replace 0/1s with human readable labels
train = train.replace({'label': {0: 'negative',
                                 1: 'positive'}})
# replace 0/1s with human readable labels
train_clean = pd.read_parquet("data/train_clean.parquet")
train_clean = train_clean.replace({'label': {0: 'negative',
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
df = (train_clean.groupby(by='label')
      .count()
      .reset_index()
      .rename(columns={'text': 'count'}))

# plot the count of pos/neg labels
plot1 = px.bar(df,
               x='label',
               y='count',
               color='label',
               title='Count of Positive and Negative Reviews',
               labels={'label': 'Movie Review Sentiment',
                       'count': 'Count of Reviews'})

# get number of tokens in each document
clean_doc_length = train_clean.tokenized_docs.apply(func=len).to_frame(name='num_tokens')

# add document length to train_clean
train_clean['num_tokens'] = clean_doc_length

# visualize document length by sentiment label
plot4 = px.histogram(train_clean,
                     x='num_tokens',
                     color='label',
                     labels={'num_tokens': 'Number of Tokens'},
                     title='Distribution of Cleaned Movie Review Length')

# intro text
app_intro = """
This app displays visualizations of the IMDB movie review dataset found [here on Kaggle](https://www.kaggle.com/columbine/imdb-dataset-sentiment-analysis-in-csv-format).
All the code for this analysis can be found on GitHub [here](https://github.com/aromatic-toast/IMDB_Sentiment_Analysis). The data displayed here is for 40,000 movie reviews
labeled as either positive or negative reviews. 
"""

markdown_text2 = """
Explore the content of positive and negative movie reviews by 
selecting from the dropdown menu. 
"""

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
            href="https://github.com/aromatic-toast/IMDB_Sentiment_Analysis",
        )
    ],
    color="dark",
    dark=True,
    id='nav-bar'
)

intro_popover = html.Div([
    dbc.Button("Click for Info",
               id="popover-target",
               color="primary",
               outline=True),
    dbc.Popover(
        [
            dbc.PopoverHeader("Introduction"),
            dbc.PopoverBody(dcc.Markdown(app_intro)),
        ],
        id="popover",
        is_open=False,
        target="popover-target"
    )
])

review_text_tooltip = dbc.Tooltip(
    markdown_text2,
    target='text-dropdown',
    placement='right'
)

############---------------BUILD THE DASH APP-----------####################
app = JupyterDash(__name__, external_stylesheets=[dbc.themes.CYBORG])

# Create server variable with Flask server object for use with gunicorn
server = app.server

app.layout = html.Div(children=[navbar,
                                dbc.Row(html.Br()),
                                intro_popover,
                                dbc.Row(html.Br()),

                                # 3 main tabs
                                dcc.Tabs([

                                    # tab to hold the summary plots
                                    dcc.Tab(id='tab1', label='Plots', children=[
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

                                    # tab to text documents
                                    dcc.Tab(id='tab2', label='Read Movie Reviews', children=[
                                        dbc.Row(html.Br()),
                                        review_text_tooltip,
                                        # dbc.Row(dbc.Col([html.H6(markdown_text2)], width=4)),
                                        dbc.Row(html.Br()),
                                        dbc.Row(dbc.Col([
                                            dcc.Dropdown(
                                                id="text-dropdown",
                                                options=[
                                                    {"label": "Review 0", "value": 0},
                                                    {"label": "Review 1", "value": 1},
                                                    {"label": "Review 2", "value": 2},
                                                ],
                                                placeholder='Select Document',
                                                value=0,
                                            )

                                        ], width=2), ),

                                        dbc.Row(html.Br()),
                                        dbc.Row([
                                            dbc.Col([dbc.Card(id='card1_content', color="primary", inverse=True)]),
                                            dbc.Col([dbc.Card(id='card2_content', color="primary", inverse=True)])
                                        ]
                                        )
                                    ]),

                                    # tab to contain a wordcloud
                                    dcc.Tab(id='tab3', label='Wordcloud', children=[
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
    Output('card1_content', component_property='children'),
    Input('text-dropdown', component_property='value'))
def get_pos_review(index):
    card_content = [
        dbc.CardHeader("Positive Reviews"),
        dbc.CardBody(
            # filter a document from the positive reviews
            children=[pos_rev[index]]
        ),
    ]
    return card_content


@app.callback(
    Output('card2_content', component_property='children'),
    Input('text-dropdown', component_property='value'))
def get_neg_review(index):
    card_content = [
        dbc.CardHeader("Negative Reviews"),
        dbc.CardBody(
            # filter a document from the positive reviews
            children=[neg_rev[index]]
        ),
    ]
    return card_content


@app.callback(
    Output("popover", "is_open"),
    [Input("popover-target", "n_clicks")],
    [State("popover", "is_open")],
)
def toggle_popover(n, is_open):
    if n:
        return not is_open
    return is_open


if __name__ == '__main__':
    app.run_server(debug=True)
