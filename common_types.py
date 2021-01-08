from typing import List

from torch import Tensor

Points = List[Tensor]  # size == [n,3]
Vertices = Tensor  # dtype == torch.long
