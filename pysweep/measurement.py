from pysweep import Namespace
from pysweep.data_storage import SpyviewStorage, BaseStorage, PandasStorage
from pysweep.sweep_object import ChainSweep


class Measurement:
    station = None

    storage_classes = {
        "spyview": {
            "storage_class": SpyviewStorage,
            "args": [],
            "kwargs": dict()
        },
        "np": {
            "storage_class": BaseStorage,
            "args": [],
            "kwargs": dict()
        },
        "pandas": {
            "storage_class": PandasStorage,
            "args": [],
            "kwargs": dict()
        }
    }

    @classmethod
    def set_station(cls, station):
        cls.station = station

    @classmethod
    def use_storage(cls, storage_class_name, **kwargs):

        if storage_class_name not in Measurement.storage_classes:
            available_classes = Measurement.storage_classes.keys()
            raise ValueError("Unknown storage class {}. Available options are {}".format(storage_class_name,
                                                                                         ",".join(available_classes)))
        storage_specs = Measurement.storage_classes[storage_class_name]

        cls.storage_class = storage_specs["storage_class"]
        cls.storage_args = storage_specs["args"]
        cls.storage_kwargs = storage_specs["kwargs"]

        for kwarg, value in kwargs.items():  # Override with user specified kwarg
            cls.storage_kwargs[kwarg] = value

    @classmethod
    def get_default_storage(cls, **user_kwargs):

        storage_kwargs = dict(cls.storage_kwargs)
        for kwarg, value in user_kwargs.items():  # Override with user specified kwarg
            storage_kwargs[kwarg] = value

        return cls.storage_class(*cls.storage_args, **storage_kwargs)

    @staticmethod
    def _make_list(value):
        if not isinstance(value, list):
            return [value]
        return value

    def __init__(self, setup, cleanup, sweep_objects, data_storage=None):

        self._setup = Measurement._make_list(setup)
        self._cleanup = Measurement._make_list(cleanup)
        self._sweep_object = ChainSweep([sweep_objects])
        self._data_storage = data_storage

        self.name = None
        self._has_run = False

    def run(self, namespace=None, **storage_kwargs):

        self._data_storage = self._data_storage or Measurement.get_default_storage(**storage_kwargs)

        if self._has_run:
            raise RuntimeError("This measurement has already run. Running twice is disallowed")
        self._has_run = True

        if namespace is None:
            namespace = Namespace()

        self._sweep_object.set_station(Measurement.station)
        self._sweep_object.set_namespace(namespace)

        for setup_function in self._setup:
            setup_function(Measurement.station, namespace)

        self._data_storage.save_json_snapshot(self.station.snapshot())

        for measurement_output in self._sweep_object:
            self._data_storage.add(measurement_output)

        self._data_storage.finalize()

        for cleanup_function in self._cleanup:
            cleanup_function(Measurement.station, namespace)

        self._sweep_object.set_namespace(None)

        return self._data_storage

