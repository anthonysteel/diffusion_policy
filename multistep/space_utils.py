from typing import Any
from gymnasium import spaces
import numpy as np

def repeated_box(box: spaces.Box, n: int) -> spaces.Box:
    """Tile a box along a new leading axis.

    Used to construct observation- and action-spaces that represent an n-stack
    of identical frames (e.g. n x RGB images or n x joint-angle vectors).

    Parameters
    ----------
    box : spaces.Box
        The base space to be repeated. Its `low`, `high`, and `dtype`
        attributes are copied.
    n : int
        Number of repetitions.

    Returns
    -------
    spaces.Box
        A new box with shape ``(n, box.shape)`` whose boundes are obtained by
        repeating `box.low` and `box.high` `n` times along axis 0.
    """
    rep = np.repeat(box.low[None, ...], n, axis=0)
    return spaces.Box(low=rep, high=np.repeat(box.high[None,...], n, axis=0),
            dtype=np.uint8)

def repeated_space(space: spaces.Space[Any], n: int) -> spaces.Space[Any]:
    """Create a new space that is n-stacked copies of an existing space.

    Parameters
    ----------
    space : spaces.Space[Any]
        Base space to be repeated. May be `spaces.Box` or `spaces.Dict` or a
        nested `spaces.Dict` of supported spaces.
    n : int
        Number of repetitions.

    Returns
    ------
    spaces.Space[Any]
        A space whose shape (for `Box`) or mapping (for `Dict`) has been
        expanded such that the first dimension is `n` copies of the original.
    """
    if isinstance(space, spaces.Box):
        return repeated_box(space, n)
    if isinstance(space, spaces.Dict):
        return spaces.Dict({k: repeated_space(v,n) for k, v in space.items()})
    raise TypeError(f"Unsupported space {type(space)}")

def stack_last(arr: list[np.ndarray], n: int) -> np.ndarray:
    """Stack the n most recent frames of a list of frames into a single array.

    Parameters
    ----------
    arr : list[np.ndarray]
        History of observation frames in chronological order
        (oldest -> newest). All arrays must share the same shape and dtype.
    n : int
        Number of frames to include in the output stack and thus the size of
        the new leading axis.

    Returns
    -------
    np.ndarray
        Array with shape ``(n, arr[0].shape)``. The last ``min(len(arr), n)``
        entries come from the tail of `arr`. If ``len(arr)<n`` the remaining
        slots are filled by repeating `arr[0]` along the leading axis
        ensuring the time dimension is always length `n`.
    """
    pad = max(0, n - len(arr))
    core = np.stack(arr[-n:], axis=0)
    if pad:
        pad_block = np.repeat(core[:1], pad, axis=0)
        core = np.concatenate([pad_block, core], axis=0)
    return core
