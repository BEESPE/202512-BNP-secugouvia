from typing import Sequence, Tuple

import numpy as np


def generate_saltpepper_masks(
    img_size: Sequence[int],
    p: float = 0.01,
) -> Tuple[np.array, np.array]:
    # ✏️ à compléter
