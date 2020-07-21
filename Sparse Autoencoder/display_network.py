import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import PIL


def display_network(A, need_normal=True, filename='weights.png'):
    """
    Visualizes generated filters.

    :param A: Input matrix with generated filters as columns.
              Each column is reshaped into a square image and visualized
              on each cell of the visualization panel.
    :param need_normal: Boolean flag for filter normalization
                        for equalizing contrast
    :param filename: File name to which filter images needs to be written.
    """
    A = A - np.average(A)

    (row, col) = A.shape
    sz = int(np.ceil(np.sqrt(row)))
    buf = 1
    n = int(np.ceil(np.sqrt(col)))
    m = int(np.ceil(col / n))

    image = np.ones(shape=(buf + m * (sz + buf), buf + n * (sz + buf)))

    k = 0
    for i in range(int(m)):
        for j in range(int(n)):
            if k >= col:
                continue

            clim = np.max(np.abs(A[:, k]))

            if need_normal:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / clim
            else:
                image[buf + i * (sz + buf):buf + i * (sz + buf) + sz, buf + j * (sz + buf):buf + j * (sz + buf) + sz] = \
                    A[:, k].reshape(sz, sz) / np.max(np.abs(A))
            k += 1

    plt.imsave(filename, image, cmap=matplotlib.cm.gray)