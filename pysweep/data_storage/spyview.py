import json
import os
from collections import defaultdict

import numpy as np
import qcodes

from pysweep.data_storage.base_storage import BaseStorage
from pysweep.utils import DictMerge


class SpyviewMetaWriter:
    def __init__(self, writer_function):

        self._writer_function = writer_function
        self._default_axis_properties = dict(max=0, min=np.inf, step=0, length=1, name="")
        self._axis_properties = defaultdict(lambda: dict(self._default_axis_properties))
        self._axis_properties["none"] = dict(max=0, min=1, step=0, length=1, name="none")

    def _update_axis_properties(self, parameter_name, parameters_values, unit):
        mn = parameters_values[0]
        mx = parameters_values[-1]

        self._axis_properties[parameter_name]["name"] = f"{parameter_name} [{unit}]"
        self._axis_properties[parameter_name]["max"] = mx
        self._axis_properties[parameter_name]["min"] = mn

        if len(parameters_values) > 1:
            self._axis_properties[parameter_name]["step"] = parameters_values[1] - mn
            self._axis_properties[parameter_name]["length"] = int(
                (mx - mn) / self._axis_properties[parameter_name]["step"]) + 1
        else:
            self._axis_properties[parameter_name]["length"] = 1

    def add(self, spyview_buffer, parameters):

        property_names = ["length", "min", "max", "name"]
        property_values = []

        for column_nr, parameter_name in enumerate(parameters):

            if parameter_name == "none" or "independent_parameter" in spyview_buffer[parameter_name]:
                if parameter_name != "none":
                    unit = spyview_buffer[parameter_name]["unit"]
                    value = sorted(set(spyview_buffer[parameter_name]["value"]))
                    self._update_axis_properties(parameter_name, value, unit)

                s = "\n".join(str(self._axis_properties[parameter_name][k]) for k in property_names)
                property_values.append(s)
                property_names = ["length", "max", "min", "name"]
            else:
                unit = spyview_buffer[parameter_name]["unit"]
                s = "\n".join([str(column_nr), f"{parameter_name} [{unit}]"])
                property_values.append(s)

        out = "\n".join(property_values)
        self._writer_function(out)


class SpyviewWriter:
    """
    Save measurement results as spyview files
    """

    def __init__(self, writer_function, meta_writer, delayed_parameters=None, max_buffer_size=1000):
        """
        Parameters
        ----------
        writer_function: callable
        delayed_parameters: list, str
            A list of delayed parameters
        max_buffer_size: int
            If the number of lines to be written to the output file exceeds this number, the buffered values will be
            written to the file system.
        """

        self._max_buffer_size = max_buffer_size
        self._delayed_parameters = delayed_parameters or []
        self._writer_function = writer_function

        self._buffer = dict()
        self._merger = DictMerge(unit="replace", value="append", independent_parameter="replace")
        self._inner_sweep_start_value = None
        self._independent_parameters = []
        self._meta_writer = meta_writer

    def _get_buffer_size(self):
        if self._buffer == dict():
            return 0

        size = np.inf
        for key in self._buffer.keys():
            value = self._get_buffer_value(key)
            if hasattr(value, "__len__"):
                this_size = len(value)
            else:
                this_size = 1

            size = min([size, this_size])
        return size

    def _find_independent_parameters(self):

        def collapse(lst):
            ary = np.array(lst)
            return np.sum(ary != np.roll(ary, -1))

        independent_collapsed = [[k, collapse(self._buffer[k]["value"])] for k in self._buffer
                                  if "independent_parameter" in self._buffer[k]]

        if len(independent_collapsed) == 0:
            raise RuntimeError("No independent parameters found. Make sure you label at least one output as an "
                               "independent parameter. Please consult the pysweep documentation at page TBD.")

        s = sorted(independent_collapsed, key=lambda el: el[1], reverse=True)
        return list(zip(*s))[0]

    def _get_buffer_value(self, key):

        if key == "empty":
            first_independent_parameter = self._independent_parameters[0]
            self._buffer["empty"] = {
                "value": [0] * len(self._buffer[first_independent_parameter]["value"]),
                "independent_parameter": True,
                "unit": ""
            }

        return self._buffer[key]["value"]

    def _write_buffer(self):

        # We will first write the independent parameters in the right order
        if len(self._independent_parameters) == 0:
            self._independent_parameters = self._find_independent_parameters()

            if len(self._independent_parameters) == 1:
                self._independent_parameters += ("empty",)

            if len(self._independent_parameters) == 2:
                self._independent_parameters += ("none",)

        all_parameters = list(self._independent_parameters)
        all_parameters.extend([param for param in self._buffer.keys() if param not in self._independent_parameters])

        buffer_values = [self._get_buffer_value(param) for param in all_parameters if param != "none"]
        inner_sweep_values = np.array(buffer_values[0])

        if self._inner_sweep_start_value is None:
            self._inner_sweep_start_value = inner_sweep_values[0]

        block_indices = inner_sweep_values == self._inner_sweep_start_value

        escapes = [{True: "\n\n", False: "\n"}[i] for i in block_indices]
        escapes[0] = ""

        lines = ["\t".join([str(ii) for ii in i]) for i in zip(*buffer_values)]
        lines = ["{}{}".format(*i) for i in zip(escapes, lines)]
        out = "".join(lines)

        self._writer_function(out)
        self._meta_writer.add(self._buffer, all_parameters)

    def add(self, dictionary):
        delayed_params_buffer = {param: {"value": []} for param in self._delayed_parameters}

        if self._buffer == {}:
            self._buffer = {k: {"value": []} for k in dictionary.keys()}

        self._buffer = self._merger.merge([dictionary, delayed_params_buffer, self._buffer])
        buffer_size = self._get_buffer_size()
        if buffer_size and buffer_size % self._max_buffer_size == 0:
            self._write_buffer()

    def finalize(self):
        self._write_buffer()

    def get_buffer(self):
        return self._buffer


class SpyviewStorage(BaseStorage):
    """
    The spyview storage module
    """
    # The storage folder can be set by the user using the "set_storage_folder" interface.
    storage_folder = ""

    # If the user does not set the storage folder, use the default one
    @staticmethod
    def default_storage_folder():
        home_path = os.path.expanduser("~")
        storage_folder = os.path.join(home_path, "data")
        return storage_folder

    @classmethod
    def set_storage_folder(cls, folder: str) ->None:
        cls.storage_folder = folder

    @classmethod
    def default_file_path(cls) ->str:
        """
        The default data output path.
        """
        if cls.storage_folder is "":
            cls.storage_folder = cls.default_storage_folder()

        data_path = os.path.join(cls.storage_folder, "{date}", "{date}_{counter}.dat")
        loc_provider = qcodes.data.location.FormatLocation(fmt=data_path)
        io = qcodes.DiskIO('.')
        file_path = loc_provider(io)

        dir_name, _ = os.path.split(file_path)
        if not os.path.exists(dir_name):
            os.makedirs(dir_name, exist_ok=True)

        return file_path

    @staticmethod
    def meta_file_path(output_file_path: str):
        dirname, filename = os.path.split(output_file_path)
        meta_file_name = filename.replace(".dat", ".meta.txt")
        return os.path.join(dirname, meta_file_name)

    @staticmethod
    def snapshot_file_path(output_file_path):
        dirname, filename = os.path.split(output_file_path)
        json_file_name = filename.replace(".dat", ".station_snapshot.json")
        return os.path.join(dirname, json_file_name)

    def __init__(self, output_file_path=None, delayed_parameters=None, max_buffer_size=10,
                 debug=False):

        self._output_file_path = output_file_path or SpyviewStorage.default_file_path()
        self._output_meta_file_path = SpyviewStorage.meta_file_path(self._output_file_path)

        self._data_folder, _ = os.path.split(self._output_file_path)

        self._max_buffer_size = max_buffer_size
        self._debug = debug

        self._meta_writer = SpyviewMetaWriter(self._meta_writer_function)
        self._writer = SpyviewWriter(
            self._writer_function,
            self._meta_writer,
            delayed_parameters=delayed_parameters,
            max_buffer_size=max_buffer_size
        )

    def _meta_writer_function(self, output):
        if self._debug:
            return

        with open(self._output_meta_file_path, "w") as fh:
            fh.write(output)

    def _writer_function(self, output):
        if self._debug:
            return

        with open(self._output_file_path, "w") as fh:
            fh.write(output)

    def output_files(self):
        return self._output_file_path, self._output_meta_file_path

    def add(self, dictionary):
        self._writer.add(dictionary)

    def output(self):
        return self._writer.get_buffer()

    def finalize(self):
        self._writer.finalize()

    def save_json_snapshot(self, snapshot):

        json_file = SpyviewStorage.snapshot_file_path(self._output_file_path)
        with open(json_file, 'w') as meta_file:
            json.dump(snapshot, meta_file, sort_keys=False, indent=4, ensure_ascii=False)



