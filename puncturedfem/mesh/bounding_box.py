import numpy as np


def get_bounding_box(x, y, tol=1e-12):
    xmin = np.min(x)
    xmax = np.max(x)
    ymin = np.min(y)
    ymax = np.max(y)
    dx = xmax - xmin
    dy = ymax - ymin
    d = np.max([dx, dy])

    if d < tol:
        return None

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
