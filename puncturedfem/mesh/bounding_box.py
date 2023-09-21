import numpy as np


def get_bounding_box(
    x: np.ndarray, y: np.ndarray, tol: float = 1e-12
) -> tuple[float, float, float, float]:
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    d = np.max([dx, dy])

    if d < tol:
        return xmin - d / 2, xmax + d / 2, ymin - d / 2, ymax + d / 2

    if dx < tol:
        x0 = xmin - d / 2
        x1 = xmin + d / 2
    else:
        x0 = xmin
        x1 = xmax

    if dy < tol:
        y0 = ymin - d / 2
        y1 = ymin + d / 2
    else:
        y0 = ymin
        y1 = ymax

    return x0, x1, y0, y1
