import numpy as np
from Array import NDArray

a = [[1,2,3], [4,5,6]]
a_t = [[1, 4], [2, 5], [3, 6]]


def test_arr_build():
    arr = NDArray(a, shape=(2, 3))
    assert arr.shape == (2, 3)
    assert (arr.numpy == np.array(a)).all()


def test_arr_t():
    arr = NDArray(a, shape=(2, 3))
    arr_t = arr.transpose()
    assert np.allclose(arr_t.numpy, np.array(a_t))
    assert id(arr.memo) == id(arr_t.memo)


def test_arr_permute():
    arr = NDArray(a, shape=(2, 3))
    arr_t = arr.permute((1, 0))
    assert np.allclose(arr_t.numpy, np.array(a_t))
    assert id(arr.memo) == id(arr_t.memo)


def test_getitem():
    arr = NDArray(a, shape=(2, 3))
    sub_arr = arr[1, :2]
    assert np.allclose(sub_arr.numpy, np.array(a[1][:2]))
    assert id(sub_arr.memo) == id(arr.memo)

    temp_a = [
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
        [
            [1, 2, 3],
            [4, 5, 6]
        ],
    ]
    arr = NDArray(temp_a, shape=(2, 2, 3))
    assert arr.shape == (2, 2, 3)
    sub_arr = arr[:, 1, :2]
    b = np.array(temp_a)[:, 1, :2]
    assert np.allclose(sub_arr.numpy, b)
    assert id(arr.memo) == id(sub_arr.memo)
