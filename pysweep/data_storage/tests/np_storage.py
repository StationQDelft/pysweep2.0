import numpy as np

import pytest

from pysweep.data_storage import NpStorage, Delayed, DataSet
from pysweep import sweep, Namespace


def test_1d():

    param = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    measure_values = np.linspace(-1, 1, 4)
    g = (i for i in measure_values)
    measure = lambda s, n: {"m": {"unit": "", "value": next(g)}}

    coordinate_values = np.array([0, 1, 2, 3])
    so = sweep(param, coordinate_values)(measure)

    store = NpStorage()

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"].flatten() == measure_values)
    assert np.all(store["m"]["x []"].flatten() == coordinate_values)


def test_2d():

    m, n = 5, 7

    param_x = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_x = np.arange(m)

    param_y = lambda s, n, v: {"y": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_y = np.arange(n)

    measure_values = np.linspace(-1, 1, m * n)
    g = (i for i in measure_values)
    measure = lambda s, n: {"m": {"unit": "", "value": next(g)}}

    so = sweep(param_x, coordinate_values_x)(sweep(param_y, coordinate_values_y)(measure))

    store = NpStorage()

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"].flatten() == measure_values)

    tmpx = store["m"]["x []"].reshape((m, n))
    tmpy = store["m"]["y []"].reshape((m, n))
    assert np.all(np.vstack([tmpy, tmpx]) == np.vstack(np.meshgrid(coordinate_values_y, coordinate_values_x)))


def test_2d_nd_measurement():
    m, n, o = 5, 7, 3

    param_x = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_x = np.arange(m)

    param_y = lambda s, n, v: {"y": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_y = np.arange(n)

    measure_values = np.arange(m * n * o).reshape((m * n, o))
    g = (i for i in measure_values)
    measure = lambda s, n: {"m": {"unit": "", "value": next(g)}}

    so = sweep(param_x, coordinate_values_x)(sweep(param_y, coordinate_values_y)(measure))

    store = NpStorage()

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"] == measure_values)

    tmpx = store["m"]["x []"].reshape((m, n))
    tmpy = store["m"]["y []"].reshape((m, n))
    assert np.all(np.vstack([tmpy, tmpx]) == np.vstack(np.meshgrid(coordinate_values_y, coordinate_values_x)))


def test_2d_nd_multi_measurement():
    m, n, o = 5, 7, 3

    param_x = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_x = np.arange(m)

    param_y = lambda s, n, v: {"y": {"unit": "", "value": v, "independent_parameter": True}}
    coordinate_values_y = np.arange(n)

    measure_values1 = np.arange(m * n * o).reshape((m * n, o))
    g = (i for i in measure_values1)
    measure1 = lambda s, n: {"m": {"unit": "", "value": next(g)}}

    measure_values2 = np.linspace(-1, 1, m)
    h = (i for i in measure_values2)
    measure2 = lambda s, n: {"n": {"unit": "", "value": next(h)}}

    so = sweep(param_x, coordinate_values_x)(measure2, sweep(param_y, coordinate_values_y)(measure1))

    store = NpStorage()

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"] == measure_values1)

    tmpx = store["m"]["x []"].reshape((m, n))
    tmpy = store["m"]["y []"].reshape((m, n))
    assert np.all(np.vstack([tmpy, tmpx]) == np.vstack(np.meshgrid(coordinate_values_y, coordinate_values_x)))

    assert np.all(store["n"]["n []"].flatten() == measure_values2)
    assert np.all(store["n"]["x []"].flatten() == coordinate_values_x)
    assert "y []" not in store["n"]


def test_delay():

    N = 16
    param = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    measure_values = np.linspace(-1, 1, N)

    def measure(s, n):
        n.count += 1
        c = n.count
        if c % 4 == 0:
            rval = DataSet(measure_values[c-4: c])
        else:
            rval = Delayed()

        return {"m": {"unit": "", "value": rval}}

    namespace = Namespace()
    namespace.count = 0
    coordinate_values = np.array(range(N))
    so = sweep(param, coordinate_values)(measure).set_namespace(namespace)

    store = NpStorage()

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"].flatten() == measure_values)
    assert np.all(store["m"]["x []"].flatten() == coordinate_values)


def test_data_set():

    xvals = np.linspace(0, 1, 10)
    yvals = np.sin(xvals)

    param = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    measure = lambda s, n, v: {"m": {"unit": "", "value": v}}

    dset = param(None, None, xvals[:5])
    dset.update(measure(None, None, yvals[:5]))

    store = NpStorage()
    store.dataset(dset)

    g = iter(yvals[5:])
    measure2 = lambda s, n: measure(s, n, next(g))

    so = sweep(param, xvals[5:])(measure2)

    for i in so:
        store.add(i)

    assert np.all(store["m"]["m []"].flatten() == yvals)
    assert np.all(store["m"]["x []"].flatten() == xvals)


def test_shape_mismatch_exception():

    param = lambda s, n, v: {"x": {"unit": "", "value": v, "independent_parameter": True}}
    measure_values = [[0, 1, 2], [3, 4, 5], [6, 7, 8, 9], [10, 11, 12]]  # The third one is longer on purpose
    g = (i for i in measure_values)
    measure = lambda s, n: {"m": {"unit": "", "value": next(g)}}

    coordinate_values = np.array([0, 1, 2, 3])
    so = sweep(param, coordinate_values)(measure)

    store = NpStorage()

    for count, i in enumerate(so):
        if count == 2:
            with pytest.raises(TypeError) as e:
                store.add(i)
            assert str(e.value) == "The shape or dtype of the measurement and/or independent parameter has suddenly " \
                                   "changed. This is not allowed"
        else:
            store.add(i)
