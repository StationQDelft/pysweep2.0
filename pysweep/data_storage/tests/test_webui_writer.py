import numpy as np

import pytest

from pysweep import sweep


from pysweep.data_storage.webui_writer import WebUIWriter


def test():
    param = lambda s, n, v: {"x": {"unit": "V", "value": v, "independent_parameter": True}}
    measure_values = np.linspace(-1, 1, 4)
    g = (i for i in measure_values)
    measure = lambda s, n: {"m": {"unit": "A", "value": next(g)}}

    coordinate_values = np.array([0, 1, 2, 3])
    so = sweep(param, coordinate_values)(measure)

    store = WebUIWriter("")

    for i in so:
        store.add(i)

    pages = store.get_pages()
    json_dict = store.page_to_json(pages["m"])

    assert json_dict["type"] == "linear"
    assert np.all(json_dict["x"]["data"] == coordinate_values)
    assert json_dict["x"]["unit"] == "V"

    assert np.all(json_dict["y"]["data"] == measure_values)
    assert json_dict["y"]["unit"] == "A"
