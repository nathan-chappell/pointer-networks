# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.8.1
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# ## Config stuff

# %autosave 0
# %matplotlib inline
# %config InlineBackend.figure_format = 'svg'

# ## Common code

# +
from pprint import pprint

from matplotlib.text import Text
import matplotlib.pyplot as plt
import numpy as np


def plot_points(points):
    plt.xlim(-0.5, 1.5)
    plt.ylim(-0.5, 1.5)
    plt.scatter(points[:, 0], points[:, 1])
    ax = plt.gca()
    # label points
    label_points(points, "p")



def label_points(points, label, y_off=0):
    ax = plt.gca()
    for i, (x, y) in enumerate(points):
        ax.add_artist(Text(x, y + y_off, f"${label}_{{{i}}}$"))


# -

# ## Convex-Hull demo

# +
from matplotlib.patches import Polygon
from convex_hull_dataset import get_points, get_verts


def get_hull(points):
    verts = get_verts(points)
    return points[verts]


def plot_hull(points):
    verts = get_hull(points)
    plt.scatter(verts[:, 0], verts[:, 1], s=4)
    # surrounding polygon
    ax = plt.gca()
    poly = Polygon(verts, color=(0.8, 0.2, 0.2, 0.5))
    ax.add_artist(poly)
    # label vertices
    label_points(verts, "v", 0.075)


def hull_demo():
    points = get_points()
    plot_points(points)
    plot_hull(points)


hull_demo()
# -

# ## Delaunay Demo

# +
from scipy.spatial import Delaunay
from matplotlib.collections import PolyCollection
import matplotlib.cm as cm


def get_delaunay(points):
    return sorted(list(map(list, Delaunay(points).simplices)))


def plot_delaunay(points):
    delaunay = get_delaunay(points)
    # pprint(delaunay)
    collection = PolyCollection(
        [points[simplex] for simplex in delaunay],
        edgecolors="black",
        cmap=cm.autumn,
    )
    collection.set_array(np.arange(len(delaunay)))
    plt.gca().add_collection(collection)
    plt.colorbar(collection)


def delaunay_demo():
    points = get_points(10)
    plot_points(points)
    plot_delaunay(points)


delaunay_demo()

