_C = None


def _lazy_import():
    global _C
    if _C is not None:
        return _C
    import torch
    from torch3d import _C as C

    _C = C
    return _C
