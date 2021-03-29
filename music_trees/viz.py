from typing import OrderedDict
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px
import plotly.io as pio

from pathlib import Path
import pandas as pd

def load_reducer(method: str):
    if method == 'umap':
        import umap
        reducer = umap.UMAP(n_components=n_components)
    elif method == 'tsne':
        from sklearn.manifold import TSNE
        reducer = TSNE(n_components=n_components)
    elif method == 'pca':
        from sklearn.decomposition import PCA
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError(f'dunno how to do {method}')

class EmbeddingVisualizer:

    def __init__(self, n_components: int=3, method='umap', title=''):
        self.n_components = n_components
        self.method = method
        self.reducer = load_reducer(method)
        self.title = title

        self.steps = []

    def add_step(self, emb, labels, symbols=None):
        proj = self.reducer.fit_transform(emb)

        if self.n_components == 2:
            df = pd.DataFrame(dict(
                x=proj[:, 0],
                y=proj[:, 1],
                label=labels
            ))
            fig = px.scatter(df, x='x', y='y', color='instrument',
                            title=self.title, symbol=symbols)

        elif self.n_components == 3:
            df = pd.DataFrame(dict(
                x=proj[:, 0],
                y=proj[:, 1],
                z=proj[:, 2],
                label=labels
            ))
            fig = px.scatter_3d(df, x='x', y='y', z='z',
                                color='instrument',
                                title=self.title, symbol=symbols)

        fig.update_traces(marker=dict(size=12,
                                      line=dict(width=1,
                                       color='DarkSlateGrey')),
                        selector=dict(mode='markers'))

import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--dir', type=str)

args = parser.parse_args()

import glob
paths = glob.glob(str(Path(args.dir) / '*.json'))
figures = {}
for fig_path in paths:
    try: 
        idx = int(str(Path(fig_path).stem)) 
    except:
        raise ValueError('the names of the .json files must be valid integers')
    with open(fig_path, 'r') as f:
        fig = pio.from_json(f.read())
    figures[idx] = fig

# sort by integer key   
figures = OrderedDict(sorted(figures.items(), key=lambda x: x[0]))

if len(figures) == 0:
    raise ValueError(f'path is empty: {args.dir}')


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    dcc.Markdown(f"""
        **embedding spaces**

        showing embeddings for {str(args.dir)}
    """),
    dcc.Graph(figure=list(figures.values())[0], id='graph-with-slider',
              style={'width': '90vh', 'height': '90vh'}),
    dcc.Slider(
        id='step-slider',
        min=list(figures.keys())[0],
        max=list(figures.keys())[-1],
        value=list(figures.keys())[0],
        marks={k: str(k) for k in list(figures.keys())},
        step=None, 
        
    )
])
    



@app.callback(
    Output('graph-with-slider', 'figure'),
    Input('step-slider', 'value'))
def update_figure(key):
    fig = figures[key]
    return fig

app.run_server(debug=True)
