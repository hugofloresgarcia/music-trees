import pandas as pd
import numpy as np
import plotly.express as px

def dim_reduce(emb, labels, symbols=None, n_components=3, method='umap', title=''):
    """
    dimensionality reduction for visualization!
    returns a plotly figure with a 2d or 3d dim reduction of ur data
    parameters:
        emb (np.ndarray): the samples to be reduced with shape (samples, features)
        labels (list): list of labels for embedding with shape (samples)
        method (str): umap, tsne, or pca
        title (str): title for ur figure
    returns:    
        fig (plotly figure): 
    """
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

    proj = reducer.fit_transform(emb)

    if n_components == 2:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            instrument=labels
        ))
        fig = px.scatter(df, x='x', y='y', color='instrument',
                         title=title, symbol=symbols)

    elif n_components == 3:
        df = pd.DataFrame(dict(
            x=proj[:, 0],
            y=proj[:, 1],
            z=proj[:, 2],
            instrument=labels
        ))
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color='instrument',
                            title=title, symbol=symbols)

    else:
        raise ValueError("cant plot more than 3 components")

    fig.update_traces(marker=dict(size=12,
                                  line=dict(width=1,
                                            color='DarkSlateGrey')),
                      selector=dict(mode='markers'))

    return fig


def plotly_fig2array(fig, dims=(1200, 700)):
    """
    convert plotly figure to numpy array
    """
    import io
    from PIL import Image
    fig_bytes = fig.to_image(format="png", width=900, height=600)
    buf = io.BytesIO(fig_bytes)
    img = Image.open(buf)
    return np.asarray(img)