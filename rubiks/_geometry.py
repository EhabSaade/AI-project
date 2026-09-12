"""Shared geometry/rotation math used to derive face-turn permutations for
an n x n x n cube (n=3 for the standard cube, n=2 for the pocket cube).

See cube.py's module docstring for why this is derived from real 3D
rotations instead of a hand-typed cycle table.
"""

import numpy as np

U, R, F, D, L, B = range(6)

_NORMALS = {
    U: (0, 1, 0),
    D: (0, -1, 0),
    R: (1, 0, 0),
    L: (-1, 0, 0),
    F: (0, 0, 1),
    B: (0, 0, -1),
}

_ROW_DIR = {
    U: (0, 0, 1),
    D: (0, 0, -1),
    F: (0, -1, 0),
    B: (0, -1, 0),
    R: (0, -1, 0),
    L: (0, -1, 0),
}
_COL_DIR = {
    U: (1, 0, 0),
    D: (1, 0, 0),
    F: (1, 0, 0),
    B: (-1, 0, 0),
    R: (0, 0, -1),
    L: (0, 0, 1),
}


def _offset(index: int, n: int) -> int:
    """Map a 0..n-1 grid coordinate to an integer position offset in [-1, 1]."""
    if n == 3:
        return index - 1
    if n == 2:
        return 2 * index - 1
    raise ValueError(f"Unsupported cube size: {n}")


def _add(a, b, scale=1):
    return (a[0] + b[0] * scale, a[1] + b[1] * scale, a[2] + b[2] * scale)


def _sticker_geometry(face, r, c, n):
    normal = _NORMALS[face]
    pos = _add(_add(normal, _ROW_DIR[face], _offset(r, n)), _COL_DIR[face], _offset(c, n))
    return pos, normal


def build_index_geometry(n: int):
    """Return (index_to_geom, geom_to_index) for an n x n x n cube's facelets."""
    index_to_geom = {}
    geom_to_index = {}
    for face in range(6):
        for r in range(n):
            for c in range(n):
                idx = face * n * n + r * n + c
                geom = _sticker_geometry(face, r, c, n)
                index_to_geom[idx] = geom
                geom_to_index[geom] = idx
    return index_to_geom, geom_to_index


def _rot_x(v, sign):
    x, y, z = v
    return (x, z, -y) if sign == -1 else (x, -z, y)


def _rot_y(v, sign):
    x, y, z = v
    return (-z, y, x) if sign == -1 else (z, y, -x)


def _rot_z(v, sign):
    x, y, z = v
    return (y, -x, z) if sign == -1 else (-y, x, z)


# For each face, clockwise-viewed-from-outside turn = rotation about the
# named axis with the given sign; only stickers whose coordinate along that
# axis equals `layer` are affected.
MOVE_DEF = {
    "U": (_rot_y, -1, 1, 1),
    "D": (_rot_y, 1, 1, -1),
    "R": (_rot_x, -1, 0, 1),
    "L": (_rot_x, 1, 0, -1),
    "F": (_rot_z, -1, 2, 1),
    "B": (_rot_z, 1, 2, -1),
}


def build_cw_permutation(face_letter: str, n: int) -> np.ndarray:
    rot_fn, sign, axis, layer = MOVE_DEF[face_letter]
    index_to_geom, geom_to_index = build_index_geometry(n)
    num_stickers = 6 * n * n
    dest = list(range(num_stickers))
    for idx, (pos, normal) in index_to_geom.items():
        if pos[axis] != layer:
            continue
        new_pos = rot_fn(pos, sign)
        new_normal = rot_fn(normal, sign)
        dest[idx] = geom_to_index[(new_pos, new_normal)]
    # dest[i] = index the sticker currently at i moves to.
    # Convert to a "gather" permutation: new_state[j] = old_state[src[j]].
    src = [0] * num_stickers
    for i, j in enumerate(dest):
        src[j] = i
    return np.array(src, dtype=np.int64)
