import json
import os
import time
import copy
from collections import OrderedDict
import numpy as np
import h5py

import qcodes

from pysweep.data_storage.base_storage import BaseStorage


class NpStorage(BaseStorage):
    
    @staticmethod
    def _analyze_record(record):
        info = {
            'dtype' : [],
            'axis_names' : [],
            'axis_units' : [],
            'datafield_names' : [],
            'datafield_units' : [],
            'inner_axes_lens' : [],
        }
        data_shape = None
        vals = []
        
        for k in record.keys():
            if record[k].get('independent_parameter', False):
                info['axis_names'].append(k)
                info['axis_units'].append(record[k]['unit'])
            else:
                info['datafield_names'].append(k)
                info['datafield_units'].append(record[k]['unit'])
            
            val = record[k]['value']
            if hasattr(val, 'dtype'):
                dt = str(val.dtype)
            else:
                dt = type(val)
            
            if type(val) == np.ndarray:
                if len(val.shape) > 0:
                    dt = str(val.shape)+dt
                    
                if data_shape is not None and val.shape != data_shape:
                    raise ValueError("All array data must have the same shape.")
                else:
                    data_shape = val.shape

            info['dtype'].append((k, dt))
            vals.append(val)
            if data_shape is not None:
                info['inner_axes_lens'] = list(data_shape)
            
        return info
    
    
    @staticmethod
    def _dict2arr(record, dtype=None):
        if dtype is None:
            dtype = NpStorage._analyze_record(record)['dtype']
        
        vals = []
        for k in record.keys():
            vals.append(record[k]['value'])
        
        return np.array([tuple(vals ), ], dtype=dtype)
    
    
    @staticmethod
    def _expand(arr):
        data = OrderedDict({})
        for f in arr.dtype.fields:
            data[f] = arr[f]
            
        return data
    
    
    @staticmethod
    def _arr2cols(arr, info, axis_order=None):
        blocksize = int(np.prod(info['inner_axes_lens']))        
        blockdata = OrderedDict({})
        dtype = []
        
        for k in info['axis_names']:
            if arr[k][0].size < blocksize:
                blockdata[k] = np.vstack(blocksize*[arr[k]]).reshape(-1, order='F')
            else:
                blockdata[k] = arr[k].reshape(-1)
            dtype.append((k, str(arr[k].dtype)))
        
        inner_axes = [np.arange(l, dtype=int) for l in info['inner_axes_lens']]
        inner_axes_grid = np.meshgrid(*inner_axes, indexing='ij')
        for i, ax in enumerate(inner_axes_grid):
            k = "Inner axis {}".format(i)
            blockdata[k] = np.concatenate(arr.size*[ax.reshape(-1)])
            dtype.append((k, 'i4'))
            
        if axis_order == 'inverted':
            blockdata = OrderedDict({k : blockdata[k] for k in reversed(blockdata)})
            dtype = dtype[::-1]
        elif axis_order is None:
            pass
        else:
            raise ValueError("Unknown axis order.")

                
        for k in info['datafield_names']:
            if arr[k][0].size < blocksize:
                blockdata[k] = np.vstack(blocksize*[arr[k]]).reshape(-1, order='F')
            else:
                blockdata[k] = arr[k].reshape(-1)
            dtype.append((k, str(arr[k].dtype)))
            
        data = np.zeros(arr.size * blocksize, dtype=dtype)
        for k in blockdata:
            data[k] = blockdata[k]
        return data

    
    def _write_spyview_data(self, col_array, prefix='data'):
        outer_axis_names = self.record_info['axis_names'][::-1]
        inner_axis_names = ["Inner axis {}".format(j) for j in \
                            range(len(self.record_info['inner_axes_lens']))][::-1]
        inner_axes = [np.arange(l, dtype=int) for l in self.record_info['inner_axes_lens']][::-1]
        
        naxes0 = len(outer_axis_names) + len(inner_axis_names)
        naxes = len(outer_axis_names) + len(inner_axis_names)
        while naxes < 3:
            outer_axis_names.append(None)
            naxes += 1
        
        isnewfile = not os.path.exists(prefix+".dat")
        with open(prefix+".dat", 'a') as f:
            for i, rec in enumerate(col_array):
                if rec[col_array.dtype.names[0]] == self.val00 and not isnewfile:
                    f.write("\n")
                
                line = "\t".join(["{}".format(rec[k]) for k in col_array.dtype.names[:naxes0]])
                if naxes0 < naxes:
                    line += ("\t" + "\t".join(["{}".format(0) for i in range(naxes-naxes0)]))
                line += ("\t" + "\t".join(["{}".format(rec[k]) for k in col_array.dtype.names[naxes0:]]))
                line += "\n"
                f.write(line)
                
        with open(prefix+".meta.txt", 'w') as f:
            iax = 0
            
            for i, k in enumerate(inner_axis_names):
                vals = np.unique(inner_axes[i])
                if iax == 1:
                    f.write("{}\n{}\n{}\n{}\n".format(vals.size, vals.max(), vals.min(), k))
                else:   
                    f.write("{}\n{}\n{}\n{}\n".format(vals.size, vals.min(), vals.max(), k))
                iax += 1

            for i, k in enumerate(outer_axis_names):
                if k is None:
                    vals = np.array([0])
                else:
                    vals = np.unique(self.data[k])

                if iax == 1:
                    f.write("{}\n{}\n{}\n{}\n".format(vals.size, vals.max(), vals.min(), str(k)))
                else:   
                    f.write("{}\n{}\n{}\n{}\n".format(vals.size, vals.min(), vals.max(), str(k)))
                iax += 1
            
            for i, k in enumerate(self.record_info['datafield_names']):
                f.write("{}\n{}\n".format(i+naxes, k))
    
         
    def __init__(self):
        self.invert_record_order = True
        self.record_info = {}
        self.data = None
        self.shape = None # need to be able to set from the outside
        self.backend = 'spyview'
        
        datestr = time.strftime("%Y-%m-%d")
        timestr = time.strftime("%H%M%S")
        self.data_folder = r"./data/{}/{}/{}".format(time.strftime("%Y-%m"), datestr, timestr)
        if not os.path.exists(self.data_folder):
            os.makedirs(self.data_folder)
            
        self.data_prefix = os.path.join(self.data_folder, "{}_{}".format(datestr, timestr))
        
        if self.backend == 'spyview':
            self.val00 = None
    
    
    def add(self, record):
        r = copy.copy(record)
        if self.invert_record_order:
            r = OrderedDict({k : r[k] for k in reversed(r)})
        
        if self.data is None:
            self.record_info = NpStorage._analyze_record(r)
            self.data = np.zeros((0,), dtype=self.record_info['dtype'])
        
        arr = NpStorage._dict2arr(r)
        self.data = np.append(self.data, arr)
        
        if self.backend == 'spyview':
            col_array = NpStorage._arr2cols(arr, self.record_info, axis_order='inverted')
            if self.val00 is None:
                self.val00 = col_array[col_array.dtype.names[0]][0]
            self._write_spyview_data(col_array, self.data_prefix)
        
    
    def output(self, *args):
        return self.data

    
    def finalize(self):
        pass

    def save_json_snapshot(self, snapshot):
        fn = self.data_prefix + "_snapshot.json"
        with open(fn, 'w') as f:
            json.dump(snapshot, f, sort_keys=False, indent=4, ensure_ascii=False)
