import numpy as np
from collections import defaultdict
from numpy.lib.recfunctions import merge_arrays

class Delayed:
    pass


class DataSet(np.ndarray):
    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)

        if len(obj.shape) == 1:
            obj = obj.reshape((-1, 1))

        return obj


class BaseStorage:

    def __init__(self, storage=None):

        if storage is None:
            self._pages = defaultdict(lambda: None)
        else:
            self._pages = storage.get_pages()

        self._delay_buffer = defaultdict(lambda: None)

    @staticmethod
    def sane_append(array1, array2):
        if array1 is None:
            return array2
        return np.append(array1, array2)

    @staticmethod
    def get_shapes(values, flatten=False):
        shapes = [v.shape if hasattr(v, "shape") else tuple() for v in values]

        if flatten:
            if len(set([shp[0] for shp in shapes])) > 1:
                raise ValueError("When flattening, the first dimension of all values must be equal")

            shapes = [shp[1:] for shp in shapes]

        return shapes

    def dict_to_array(self, dictionary):

        units = [d["unit"] for d in dictionary.values()]
        field_names = ["{} [{}]".format(*nu) for nu in zip(dictionary.keys(), units)]

        values = [np.atleast_1d(d["value"]) for d in dictionary.values()]
        flatten = any([isinstance(v, DataSet) for v in values])

        shapes = self.get_shapes(values, flatten=flatten)

        types = [str(shp) + str(i.dtype) for shp, i in zip(shapes, values)]
        dtype = list(zip(field_names, types))

        if flatten:
            rval = np.array([tuple(i) for i in zip(*values)], dtype=dtype)
        else:
            rval = np.array([tuple(values)], dtype=dtype)

        return rval

    def add(self, record):

        independents = [(name, value) for name, value in record.items() if value.get('independent_parameter', False)]
        independents = self.dict_to_array(dict(independents))

        dependents = [(name, value) for name, value in record.items() if not value.get('independent_parameter', False)]

        for name, parameter_record in dependents:

            dependent = self.dict_to_array({name: parameter_record})

            if isinstance(parameter_record["value"], Delayed):
                self._delay_buffer[name] = self.sane_append(self._delay_buffer[name], independents)
                continue

            elif isinstance(parameter_record["value"], DataSet) and name in self._delay_buffer:
                independents = np.append(self._delay_buffer[name], independents)
                del self._delay_buffer[name]

            page = merge_arrays([independents, dependent], flatten=True, usemask=False)

            try:
                self._pages[name] = self.sane_append(self._pages[name], page)
            except TypeError:
                raise TypeError("The shape or dtype of the measurement and/or independent parameter has suddenly "
                                "changed. This is not allowed")

    def dataset(self, dset):

        dset = dict(dset)  # Make a copy
        for record in dset.values():
            record["value"] = DataSet(record["value"])

        self.add(dset)

    def output(self, item):
        return self[item]

    def __getitem__(self, item):
        if item not in self._pages:
            raise KeyError("{} not found. Possible values are {}".format(item, ",".join([str(i) for i in
                                                                                         self._pages.keys()])))
        return self._pages.__getitem__(item)

    def finalize(self):
        pass

    def save_json_snapshot(self, snapshot):
        self.snapshot = snapshot

    def get_pages(self):
        return self._pages
