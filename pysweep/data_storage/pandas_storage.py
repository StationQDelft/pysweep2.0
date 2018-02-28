import numpy as np
import pandas as pd

from pysweep.data_storage import BaseStorage


class PandasStorage(BaseStorage):

    def __getitem__(self, item):
        data_frame = super().__getitem__(item)
        dtype = data_frame.dtype

        if any([dtype[n].subdtype[1] != (1,) for n in dtype.names]):
            raise TypeError("Cannot convert frames containing multi-dimensional arrays to a Pandas frame")

        new_dtype = [(n, dtype[n].subdtype[0]) for n in dtype.names]
        new_data_frame = np.ndarray.view(data_frame, dtype=new_dtype)
        return pd.DataFrame(new_data_frame)

    def write(self):
        pass  # For now

    def save_json_snapshot(self, snapshot):
        pass  # for now