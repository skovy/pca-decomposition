import sys
import plotly
from plotly.tools import FigureFactory
from plotly.graph_objs import Scatter, Layout
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from src.track_data import TrackData

samples = 100 # the number of data samples to graph
index_of_attribute_to_classify = 0
indices = [] # indices color distinctly

attrs = ["danceability", "loudness", "energy", "speechiness", "liveness", "acousticness", "instrumentalness"]

if len(sys.argv) > 1:
    samples = int(sys.argv[1])
if len(sys.argv) > 2:
    index_of_attribute_to_classify = int(sys.argv[2])
start = 3
while len(sys.argv) > start:
    indices.append(int(sys.argv[start]))
    start += 1

indices.sort()
attr = attrs[index_of_attribute_to_classify]

td = TrackData().retrieve_data(samples, index_of_attribute_to_classify)
data = []
for i, sample in enumerate(td):
    data.append([])
    for j, col in enumerate(sample[2:]):
        if index_of_attribute_to_classify == j:
            continue

        if col != None:
            data[i].append(float(col))
        else:
            data[i].append(0)

X = normalize(np.array(data), norm = 'l2', axis = 1)
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)

x_danceable = []
y_danceable = []
text_danceable = []
x_non_danceable = []
y_non_danceable = []
text_non_danceable = []
indices_x = []
indices_y = []
indices_text = []

for i, sample in enumerate(X_r):
    is_an_index = False
    for index in indices:
        if index == td[i][1]:
            is_an_index = True
            break

    if is_an_index:
        indices_x.append(sample[0])
        indices_y.append(sample[1])
        indices_text.append(td[i][0])
    else:
        if td[i][index_of_attribute_to_classify + 2] > 0.5:
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
        name = attr
    ),
    Scatter(
        x = x_non_danceable,
        y = y_non_danceable,
        mode = "markers",
        marker = dict(size = 10),
        text = text_non_danceable,
        name = "non-" + attr
    )
]

for i, j in enumerate(indices):
    data.append(
        Scatter(
            x = [indices_x[i]],
            y = [indices_y[i]],
            mode = "markers",
            marker = dict(size = 10),
            text = [indices_text[i]],
            name = "index_" + str(j)
        )
    )

plotly.offline.plot(
    {
        "data": data,
        "layout": Layout(
            title = attr
        )
    },
    filename = 'basic-scatter'
)
