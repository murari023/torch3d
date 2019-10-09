_C = None


def _lazy_import():
    global _C
    if _C is not None:
        return _C

    import torch
    import torch3d._C as C
    _C = C
    return _C
