import plotly
from plotly.tools import FigureFactory
from plotly.graph_objs import Scatter, Layout
import numpy as np
from sklearn.decomposition import PCA
from src.track_data import TrackData

td = TrackData().retrieve_data(460)
data = []
for i, sample in enumerate(td):
    data.append([])
    for j, col in enumerate(sample[2:]):
        data[i].append(0)
        if col != None:
            data[i][j] = float(col)

X = np.array(data)
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)
print(X_r)
target_names = ['a', 'b', 'c']


x_danceable = []
y_danceable = []
text_danceable = []
x_non_danceable = []
y_non_danceable = []
text_non_danceable = []
for i, sample in enumerate(X_r):
    if td[i][1] > 0.5:
        x_danceable.append(sample[0])
        y_danceable.append(sample[1])
        text_danceable.append(td[i][0])
    else:
        x_non_danceable.append(sample[0])
        y_non_danceable.append(sample[1])
        text_non_danceable.append(td[i][0])

data = [
    Scatter(
        x = x_danceable,
        y = y_danceable,
        mode = "markers",
        marker = dict(size = 10),
        text = text_danceable,
        name = "danceable"
    ),
    Scatter(
        x = x_non_danceable,
        y = y_non_danceable,
        mode = "markers",
        marker = dict(size = 10),
        text = text_non_danceable,
        name = "non_danceable"
    )
]

plotly.offline.plot(
    {
        "data": data,
        "layout": Layout(
            title = "Danceability"
        )
    },
    filename = 'basic-scatter'
)
