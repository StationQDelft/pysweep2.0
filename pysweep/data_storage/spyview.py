import numpy as np
import os
import json
import time
import re
from collections import defaultdict

import qcodes

from pysweep.data_storage.base_storage import BaseStorage


class FileWriter:
    def __init__(self, file_path, mode="w"):
        self._file_path = file_path
        self._mode = mode

    def write(self, out_string):
        with open(self._file_path, self._mode) as fh:
            fh.write(out_string)

    @property
    def file_path(self):
        return self._file_path


class SpyviewMetaWriter:
    def __init__(self, writer):
        self._writer = writer

    @staticmethod
    def find_axis_properties(array):
        axis = np.unique(array)
        return dict(
            min=np.min(axis),
            max=np.max(axis),
            length=len(axis)
        )

    def write(self, page):
        property_names = ["length", "min", "max", "name"]
        property_values = []

        independent_parameters = list(page.dtype.names[:-1])

        if len(independent_parameters) == 2:
            independent_parameters.append("none")

        for axis_name in independent_parameters:

            if axis_name != "none":
                axis_properties = self.find_axis_properties(page[axis_name])
            else:
                axis_properties = dict(max=0, min=1, length=1)

            axis_properties["name"] = axis_name

            s = "\n".join(str(axis_properties[k]) for k in property_names)
            property_values.append(s)
            property_names = ["length", "max", "min", "name"]

        axis_name = page.dtype.names[-1]
        s = "\n".join(["3", axis_name])
        property_values.append(s)

        out = "\n".join(property_values)
        self._writer.write(out)


class SpyviewWriter:
    def __init__(self, writer, meta_writer):
        self._writer = writer
        self._meta_writer = meta_writer

    @staticmethod
    def insert_field(array, position, field_name, field_dtype):

        names = array.dtype.names

        new_dtype = [array.dtype[n] for n in names]
        new_dtype.insert(position, field_dtype)

        new_names = list(names)
        new_names.insert(position, field_name)

        new_array = np.zeros(array.shape, dtype=list(zip(new_names, new_dtype)))

        for name in names:
            new_array[name] = array[name]

        return new_array

    def write(self, page):
        params = list(page.dtype.names)
        if len(params) == 2:
            page = self.insert_field(page, 1, "empty", np.dtype((int, (1,))))
            params.insert(1, "empty")

        inner = params[0]
        inner_sweep_values = page[inner]
        block_indices = (inner_sweep_values == inner_sweep_values[0]).flatten()
        escapes = [{True: "\n\n", False: "\n"}[i] for i in block_indices]
        escapes[0] = ""

        lines = ["\t".join([str(i[0]) for i in p[0]]) for p in list(zip(page))]
        lines = ["{}{}".format(*i) for i in zip(escapes, lines)]
        out = "".join(lines)

        out = "\n".join(["# {}".format(i) for i in params]) + "\n" + out
        self._writer.write(out)
        self._meta_writer.write(page)

    @property
    def file_path(self):
        return self._writer.file_path


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

    @staticmethod
    def increment_file_number(file_path, increment=1):
        ss = "_([0-9]*).dat"
        result = re.search(ss, file_path)
        file_num = int(result.groups()[0])
        sub = "_{:0>3}.dat".format(file_num + increment)
        return re.sub(ss, sub, file_path)

    def __init__(self, write_delay=5):
        super().__init__()
        self._spywriters = defaultdict(self.create_spyview_writer)

        self._last_write_action = time.time()
        self._write_delay = write_delay
        self._base_file_path = SpyviewStorage.default_file_path()
        self._n_spyview_files = 0

    def create_spyview_writer(self):
        file_path = self.increment_file_number(self._base_file_path, self._n_spyview_files)
        file_writer = FileWriter(file_path)
        self._n_spyview_files += 1

        meta_file_path = self.meta_file_path(file_path)
        meta_file_writer = FileWriter(meta_file_path)
        meta_writer = SpyviewMetaWriter(meta_file_writer)

        if self._base_file_path is None:
            self._base_file_path = file_path

        return SpyviewWriter(file_writer, meta_writer)

    def add(self, dictionary):
        super().add(dictionary)

        if time.time() - self._last_write_action >= self._write_delay:
            self._write()

    def _write(self):
        for param in self._pages.keys():
            page = self.output(param)
            self._spywriters[param].write(page)

    def finalize(self):
        self._write()

    def save_json_snapshot(self, snapshot):

        json_file = SpyviewStorage.snapshot_file_path(self._base_file_path)
        with open(json_file, 'w') as meta_file:
            json.dump(snapshot, meta_file, sort_keys=True, indent=4, ensure_ascii=False)
