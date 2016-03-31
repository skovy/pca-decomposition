import plotly
from plotly.tools import FigureFactory
from plotly.graph_objs import Scatter, Layout
import numpy as np
from sklearn.decomposition import PCA

X = np.array([[6, 5, 6, -1, -1, -1], [6, 5, 6, -2, -1, -1], [6, 5, 6, -3, -2, -1], [6, 5, 6, 1, 1, 1], [6, 5, 6, 2, 1, 1], [6, 5, 6, 3, 2, 1]])
pca = PCA(n_components=2)
X_r = pca.fit_transform(X)
print(X_r)
target_names = ['a', 'b', 'c']


x = []
y = []
for sample in X_r:
    x.append(sample[0])
    y.append(sample[1])

data = [
    Scatter(
        x=x,
        y=y,
        mode="markers",
        marker = dict(size = 10)
    )
]

plotly.offline.plot(
    {
        "data": data,
        "layout": Layout(
            title = "Plot"
        )
    },
    filename = 'basic-scatter'
)
