from dataclasses import dataclass
from typing import List, Tuple, Optional, Union, cast
import dataclasses
import pickle

from matplotlib.patches import Polygon
from scipy.spatial import ConvexHull
from torch import tensor, Tensor
from torch.utils.data import TensorDataset, DataLoader, Dataset, ConcatDataset
from tqdm import tqdm
import numpy as np
import torch

from common_types import Points, Vertices


def get_points(n=20) -> np.array:
    return np.random.random((n, 2))


def get_verts(points) -> np.array:
    hull = ConvexHull(points)
    verts = hull.vertices
    verts = np.roll(verts, -np.argmin(verts))
    # print(verts)
    return verts


def encode_verts(verts: np.array, size: int) -> np.array:
    """one-hot encode the sequence of vertices...
    position 0,1 are the beginning/ end symbols
    """
    # inspired by
    # https://stackoverflow.com/questions/29831489/convert-array-of-indices-to-1-hot-encoded-numpy-array
    _verts = np.concatenate(([0], verts + 2, [1]))
    one_hot = np.zeros((len(_verts), size + 2))
    one_hot[np.arange(len(_verts)), _verts] = 1
    return one_hot


default_sizes = [5, 10, 15]


@dataclass
class ConvexHullSample:
    points: Points
    # tensor of indices
    vertices: Vertices

    @staticmethod
    def create_samples(n: int, size: int) -> List["ConvexHullSample"]:
        samples: List["ConvexHullSample"] = []
        print(f"creating {n} samples of size {size}")
        for _ in tqdm(range(n)):
            points = get_points(size)
            vertices = get_verts(points)
            _points = np.pad(points, [(0, 0), (0, 1)])
            samples.append(
                ConvexHullSample(
                    points=[tensor(point, dtype=torch.float32) for point in _points],
                    vertices=tensor(vertices, dtype=torch.long),
                )
            )
        return samples


class ConvexHullDataset(list, Dataset[ConvexHullSample]):
    def __add__(  # type: ignore[override]
        self, other: Dataset[ConvexHullSample]
    ) -> ConcatDataset[ConvexHullSample]:
        return super(list, self).__add__(other)


def create_dataset(
    n: int, sizes=default_sizes, filename: Optional[str] = None
) -> ConvexHullDataset:
    """Create dataset with n samples"""
    dataset: ConvexHullDataset = ConvexHullDataset()
    for size in sizes:
        dataset.extend(ConvexHullSample.create_samples(n, size))
    if filename is not None:
        with open(filename, "wb") as file:
            pickle.dump(dataset, file)
        print(f"saved {n} samples to {filename}")
    return dataset


def get_dataloader(
    data_source: Union[str, ConvexHullDataset], seed: int = 0
) -> DataLoader:
    torch.manual_seed(seed)
    if isinstance(data_source, str):
        with open(data_source, "rb") as file:
            data_source = pickle.load(file)
    data_dict_list = [dataclasses.asdict(data_sample) for data_sample in data_source]
    # return DataLoader(cast(Dataset, data_source), batch_size=1, shuffle=True)
    return DataLoader(cast(Dataset, data_dict_list), batch_size=1, shuffle=True)
