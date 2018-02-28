import numpy as np
import re
import json

from pysweep.data_storage.base_storage import BaseStorage


class WebUIWriter(BaseStorage):

    def __init__(self, output_file, storage=None, write_delay=5):
        super().__init__(storage=storage, write_delay=write_delay)
        self.output_file = output_file

    @staticmethod
    def page_to_json(page):
        """
        Parameters
        ----------
        page: np.ndarray
        """
        dtype = page.dtype
        parameters = dtype.names

        if any([dtype[n].subdtype[1] != (1,) for n in parameters]):
            raise TypeError("Cannot convert frames containing multi-dimensional arrays to a WebUI plot")

        try:
            plot_type = {2: "linear", 3: "heatmap"}[len(parameters)]
        except KeyError:
            raise TypeError("The page supplied represents neither 2D or 3D data")

        rval = {"type": plot_type}

        for count, (axis, parameter) in enumerate(zip("xyz", parameters)):

            name, unit = re.search("(.*) \[(.*)\]", parameter).groups()

            rval[axis] = {
                "data": list(page[parameter].flatten()),
                "full_name": name,
                "is_setpoint": count < len(parameters) - 1,
                "name": name,
                "unit": unit
            }

        return rval

    def write(self):
        pages = self.get_pages()
        if len(pages) > 1:
            raise NotImplementedError("There is no support yet for exporting multiple data sets for WebUI plotting"
                                      "as the underlying frame work does not support this (yet)")
        page = pages[0]
        json_dict = WebUIWriter.page_to_json(page)

        with open(self.output_file, "w") as fh:
            json.dump(json_dict, fh)

    def save_json_snapshot(self, snapshot):
        pass

