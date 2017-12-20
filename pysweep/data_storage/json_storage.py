"""
Initial work on the JSON storage class. Note that for now we are accumulating data in a dictionary without writing
anything to file yet
"""

from pysweep.data_storage.base_storage import BaseStorage
from pysweep.utils import DictMerge


class JSONStorage(BaseStorage, DictMerge):
    def __init__(self, **strategy):
        super().__init__(**strategy)
        self.buffer = []
        self._unique_names = set()
        self.snapshot = None

    def add(self, dictionary):

        for key in dictionary.keys():
            self._unique_names.add(key)

        self.buffer.append(dict(dictionary))

    def output(self, name):

        if name not in self._unique_names:
            raise ValueError("Parameter {} unknown".format(name))

        d = dict()
        for ibuffer in self.buffer:
            if name in ibuffer:
                d = self.merge([ibuffer, d])
        return d

    def keys(self):
        return list(self._unique_names)

    def finalize(self):
        pass  # TODO: Left off here

    def save_json_snapshot(self, snapshot):
        self.snapshot = snapshot
