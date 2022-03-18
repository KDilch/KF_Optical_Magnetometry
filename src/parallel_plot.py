import matplotlib
import matplotlib.pyplot as plt
import os
import shutil
import functools
from multiprocessing import get_context
import io
import numpy as np
from PIL import Image


CACHE_DIR = ".figcache"
DPI = 96


# noinspection PyUnresolvedReferences
def _parallel_plot_worker(args, plot_fn, fig_size, in_memory, preprocessing_fn=None, dpi=96, pad_inches=0.1):
    index, data = args

    if preprocessing_fn is not None:
        data = preprocessing_fn(data)

    fig = plt.figure(figsize=fig_size)
    matplotlib.font_manager._get_font.cache_clear()  # necessary to reduce text corruption artifacts
    axes = plt.axes()

    plot_fn(data, fig, axes)

    if not in_memory:
        path = f"{CACHE_DIR}/{index}.temp.png"
        plt.savefig(path, format="png", bbox_inches="tight", dpi=dpi, pad_inches=pad_inches)
        plt.close()
        return index, path
    else:
        buf = io.BytesIO()
        fig.savefig(buf, format="png", bbox_inches="tight", dpi=dpi, pad_inches=pad_inches)
        buf.seek(0)
        img = np.array(Image.open(buf))
        buf.close()
        plt.close()
        return index, img


def _make_subplots(data, plot_fn, n_rows, n_cols, grid_cell_size, total=None, preprocessing_fn=None, switch_axis=False,
                   show_progress=True, in_memory=True, mp_context=None, dpi=96, pad_inches=0.1, max_workers=-1):

    if total is None:
        total = len(data)

    worker_func = functools.partial(_parallel_plot_worker, plot_fn=plot_fn, fig_size=grid_cell_size,
                                    in_memory=in_memory, preprocessing_fn=preprocessing_fn, dpi=dpi,
                                    pad_inches=pad_inches)

    subplot_results = np.empty((n_rows, n_cols), dtype=object)

    workers = min(total, os.cpu_count())
    if max_workers != -1:
        workers = min(workers, max_workers)

    with get_context(mp_context).Pool(workers) as pool:

        iterator = pool.imap_unordered(worker_func, zip(range(total), data))
        if show_progress:
            from tqdm.auto import tqdm
            iterator = tqdm(iterator, total=total)

        for index, img in iterator:

            if not in_memory:
                img = plt.imread(img)

            if switch_axis:
                c = int(index / n_rows)
                r = index % n_rows
            else:
                c = index % n_cols
                r = int(index / n_cols)

            subplot_results[r, c] = img

    return subplot_results


def parallel_plot(plot_fn, data, grid_shape, total=None, preprocessing_fn=None, col_labels=None, row_labels=None,
                  grid_cell_size=(6, 12), switch_axis=False, cleanup=True, show_progress=True, in_memory=True,
                  mp_context=None, max_workers=-1):
    """
    Generate a grid of plots, where each plot inside the grid is generated by another process,
    effectively allowing parallel plot generation.

    :param plot_fn: Plot function that will be called from the process context. Lambda expressions are not supported.
    :param data: Iterable data with length of rows * cols.
    :param grid_shape: Shape of the grid as (rows, cols) tuple. For a horizontal list provide (N, 1) and for a vertical
    list provide (1, N).
    :param total: Length of the data. Must be provided if the passed data length cannot be accessed by calling len()
    e.g. on generators
    :param preprocessing_fn: Optional preprocessing function that is called from the process context on the data chunk
    before plotting. Lambda expressions are not supported.
    :param col_labels: Optional list of column labels.
    :param row_labels: Optional list of row labels.
    :param grid_cell_size: Size of each cell (subplot) as (width, height) tuple. This has a direct impact on the parent
    plot size.
    :param switch_axis: If false the grid will be populated from left to right, top to bottom. Otherwise, it will be
    populated top to bottom, left to right.
    :param cleanup: If true, the generated cache directory will be deleted before finishing. Can be useful for
    debugging. Only active when in_memory is False.
    :param show_progress: If true, shows a progressbar of the plotting. Requires the tqdm module.
    :param in_memory: If true (Default) will pass images directly to main instead of writing to drive.
    :param mp_context: str that identifies which spawn-method multiprocessing should use. OS- dependant and typically
    one of "fork", "forksever", or "spawn". Leave None for system default.
    :param max_workers: Maximum number of worker processes to spawn. Leave at -1 to use as many as possible.
    :return: fig, axes of the parent plot.
    """

    if not in_memory and not os.path.exists(CACHE_DIR):
        os.mkdir(CACHE_DIR)

    n_rows = grid_shape[0]
    n_cols = grid_shape[1]

    full_fig_size = (n_cols * grid_cell_size[0], n_rows * grid_cell_size[1])

    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=full_fig_size)

    if col_labels is not None:
        for label, ax in zip(col_labels, axes[0]):
            ax.set_title(label, loc="center", wrap=True)

    if row_labels is not None:
        for label, ax in zip(row_labels, axes[:, 0]):
            ax.set_ylabel(label, loc="center", wrap=True)

    for ax in axes.ravel():
        ax.get_xaxis().set_ticks([])
        ax.get_yaxis().set_ticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    subplots = _make_subplots(data, plot_fn, n_rows, n_cols, grid_cell_size, total, preprocessing_fn, switch_axis,
                              show_progress, in_memory, mp_context, dpi=DPI, max_workers=max_workers)

    for ax, img in zip(axes.ravel(), subplots.ravel()):
        ax.imshow(img)

    plt.subplots_adjust(hspace=0, wspace=0)

    if not in_memory and cleanup:
        shutil.rmtree(CACHE_DIR)

    return fig, axes