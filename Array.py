"""
1. NDArray only support int, float dtype.
"""
from functools import reduce
from operator import mul
import numpy as np
import math


def get_elements(seq, container):
    """Sequential all elements in seq to container"""
    for it in seq:
        if isinstance(it, list):
            get_elements(it, container)
        else:
            assert isinstance(it, (int, float))
            container.append(it)


def compact_strides(shape):
    """ Utility function to compute compact strides """
    stride = 1
    res = []
    for i in range(1, len(shape) + 1):
        res.append(stride)
        stride *= shape[-i]
    return tuple(res[::-1])


def prod(shape: tuple):
    """Return the prod of shape"""
    return reduce(mul, shape)


def to_numpy(a, shape, strides, offset, datatype_size):
    return np.lib.stride_tricks.as_strided(
        a[offset:], shape, tuple([s * datatype_size for s in strides])
    )


class Memory:
    """
    Simulate low-level contiguous storage of ndarray
    """

    def __init__(self, data=None, size=None):
        """(data, size) at least one is given.
        Args:
            data: list of lists
            size: the number of elements (float or int or ...) in data
        """
        assert data or size is not None

        if data is not None:
            self.data = []
            get_elements(data, self.data)
            self.size = len(self.data)
        else:
            assert size is not None
            self.data = [0] * size
            self.size = size

    def __getitem__(self, item):
        return self.data[item]

    def __eq__(self, other):
        return self.data == other.data


class NDArray:
    """
    NDArray implementation.
    """

    def __init__(self, data, shape, dtype=None):
        """
        Args:
            data: list of lists or list of int/float
            shape: tuple of int
            dtype: int or float. Only those two are supported
        """
        if data is not None:
            self.memo = Memory(data=data)
        else:
            self.memo = Memory(size=prod(shape))

        self.shape = shape
        self.ndim = len(self.shape)
        self.strides = compact_strides(shape)

        if dtype is not None:
            self.dtype = dtype
        else:
            self.dtype = type(self.memo[0])

        self.offset = 0

    def process_slice(self, sl, dim):
        """ Convert a slice to an explicit start/stop/step """
        start, stop, step = sl.start, sl.stop, sl.step
        if start == None:
            start = 0
        if start < 0:
            start = self.shape[dim]
        if stop == None:
            stop = self.shape[dim]
        if stop < 0:
            stop = self.shape[dim] + stop
        if step == None:
            step = 1

        # we're not gonna handle negative strides and that kind of thing
        assert stop > start, "Start must be less than stop"
        assert step > 0, "No support for  negative increments"
        return slice(start, stop, step)

    @property
    def numpy(self):
        datatype_size = np.dtype(self.dtype).itemsize
        return to_numpy(self.memo.data, self.shape, self.strides, self.offset, datatype_size)

    def create_view(self, shape, strides, offset=0):
        """Only create a view of the NDArray, not allocate new memory for data"""
        arr = NDArray.__new__(NDArray)
        arr.memo = self.memo
        arr.shape = shape
        arr.dtype = self.dtype
        arr.strides = strides
        arr.offset = offset
        return arr

    def __getitem__(self, idxs):
        if not isinstance(idxs, tuple):
            idxs = (idxs,)
        idxs = tuple(
            [
                self.process_slice(s, i) if isinstance(s, slice) else slice(s, s + 1, 1)
                for i, s in enumerate(idxs)
            ]
        )
        assert len(idxs) == self.ndim, "Need indexes equal to number of dimensions"

        new_shape = []
        for idx in idxs:
            new_shape.append(math.ceil((idx.stop - idx.start) / idx.step))
        new_shape = tuple(new_shape)

        new_strides = []
        offset = 0
        for i in range(len(idxs)):
            new_strides.append(self.strides[i] * idxs[i].step)
            offset += self.strides[i] * idxs[i].start
        new_strides = tuple(new_strides)

        return self.create_view(new_shape, new_strides, offset)

    def __repr__(self):
        datatype_size = np.dtype(self.dtype).itemsize
        return f"NDArray (\n{to_numpy(self.numpy, self.shape, self.strides, self.offset, datatype_size).__str__()}, " \
               f"dtype={self.dtype} )\n"

    def reshape(self, new_shape: tuple):
        if prod(new_shape) != prod(self.shape):
            raise Exception("prod(new_shape) != prod(self.shape)")
        else:
            return self.create_view(new_shape, compact_strides(new_shape))

    def permute(self, new_axes: tuple):
        if len(new_axes) != len(self.shape):
            raise ValueError("len(new_axes) != len(self.shape)")

        new_shape = []
        new_strides = []
        for ax in new_axes:
            new_shape.append(self.shape[ax])
            new_strides.append(self.strides[ax])

        return self.create_view(tuple(new_shape), tuple(new_strides))

    def transpose(self):
        return self.permute((1, 0))

    def __sub__(self, other):
        ...

    def __add__(self, other):
        ...

    def __mul__(self, other):
        ...

    def __matmul__(self, other):
        ...

