
import typing as ty
import torch


class CastingError(Exception):
    def __init__(self, inp_dtype : ty.AnyStr, e : Exception):
        super().__init__(f"""
            Casting from type {inp_dtype} to {torch.int32} cannot be executed.
            Refer to https://github.com/JafarBadour/Parallel-Dynamic-Sparse-Training
            
            stack {e.__traceback__}
        """)
