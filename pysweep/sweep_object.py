"""
Sweep object definitions
"""
import collections
from collections import OrderedDict
import copy
import itertools
import time

import qcodes


class BaseSweepObject:
    """
    A sweep object is defined as follows:
    1) It is iterable and looping over a sweep object shall result in independent measurement parameters being set
    at each iteration
    2) At each iteration a dictionary is returned containing information about the parameters set and measurements that
    have been performed.
    """

    @staticmethod
    def parameter_log_format(parameter, value, setting=True):
        """
        The standard method of logging the setting or retrieving a parameter value

        Parameters
        ----------
        parameter: qcodes.StandardParameter
        value: float
            Either the set or the get value of the parameter. As the sole responsibility of this function is to log
            the setting or retrieving of parameter values, the setting and getting should have been done on the calling
            side.
        setting: bool, optional (True)
            Are we setting a parameter or retrieving it? If we are setting it, the the parameters is coupled to an
            independent measurement parameter. If we are retrieving it (measuring it), which means it is an dependent
            parameter
        """
        if parameter._instrument is not None:  # TODO: Make a QCoDeS pull request to access this through a public
            # interface
            label = "{}_{}".format(parameter._instrument.name, parameter.label)
        else:
            label = parameter.label

        log_dict = {label: {"unit": parameter.unit, "value": value}}
        if setting:
            log_dict[label]["independent_parameter"] = True

        return OrderedDict(log_dict)

    def __init__(self):
        # The following attributes are determined when we begin iteration...
        self._param_setter = None  # A "param_setter" is an iterator which, when "next" is called a new value of the
        # independent parameter is set.
        self._station = None  # A QCoDeS station which allows us to access measurement instruments
        self._namespace = None  # The namespace allows different measurement functions to share a memory space

    def _setter_factory(self):
        """
        When called, this method returns the param setter iterable appropriate for this measurement
        """
        raise NotImplementedError("Please subclass BaseSweepObject")

    def __iter__(self):
        self._param_setter = self._setter_factory()
        return self

    def __next__(self):
        if self._param_setter is None:
            self._param_setter = self._setter_factory()
        return next(self._param_setter)

    def __call__(self, *objects):
        chained_sweep_object = ChainSweep(objects)
        return NestedSweep([self, chained_sweep_object])

    def set_station(self, station):
        """
        Make a station available to measurements being performed

        Parameters
        ----------
        station: qcodes.station

        Returns
        -------
        self
        """
        if station is not None:
            self._station = station
        return self

    def set_namespace(self, namespace):
        """
        Make a namespace available to measurements being performed

        Parameters
        ---------
        namespace: pysweep.Namespace

        Returns
        -------
        self
        """
        if namespace is not None:
            self._namespace = namespace
        return self


class CompoundSweep(BaseSweepObject):
    """
    A compound sweep object is composed to multiple sweep objects. Examples are a nested sweep or a zipped sweep.
    This is also an abstract class.

    Parameters
    ----------
    sweep_objects: list, BaseSweepObject
    """

    def __init__(self, sweep_objects):
        self._sweep_objects = []

        for so in sweep_objects:
            if isinstance(so, list):
                so = ChainSweep(so)
            elif isinstance(so, tuple):
                so = NestedSweep(so)
            elif isinstance(so, qcodes.instrument.parameter.Parameter):
                so = ParameterWrapper(so)
            elif isinstance(so, BaseSweepObject):
                so = so
            elif callable(so):
                so = FunctionWrapper(so)

            self._sweep_objects.append(so)

        super().__init__()

    def set_station(self, station):
        """
        We need to pass on the station to the sweep objects in the sub sweep objects

        Parameter
        ---------
        station: qcodes.Station
        """
        self._station = station
        for so in self._sweep_objects:
            so.set_station(station)
        return self

    def set_namespace(self, namespace):
        """
        We need to pass on the namespace to the sweep objects in the sub sweep objects

        Parameter
        ---------
        namespace: pysweep.Namespace
        """
        self._namespace = namespace
        for so in self._sweep_objects:
            so.set_namespace(namespace)
        return self

    def _setter_factory(self):
        """
        When called, this method returns the param setter iterable appropriate for this measurement
        """
        raise NotImplementedError("Please subclass BaseSweepObject")

# --------------Sweep Subclasses ----------------------------------------------------------------------------


class IteratorSweep(BaseSweepObject):
    """
    Sweep independent parameters by unrolling an iterator. This class is useful if we have "bare" parameter set iterator
    and need to create a proper sweep object as defined in the BaseSweepObject docstring. See the "NestedSweep"
    subclass for an example.

    Parameters
    ----------
    iterator_function: callable
        A callable with no parameters, returning an iterator. Unrolling this iterator has the
        effect of setting the independent parameters.
    """
    def __init__(self, iterator_function):
        self._iterator_function = iterator_function
        super().__init__()

    def _setter_factory(self):
        for value in self._iterator_function():
            yield value


class ParameterSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and setting a QCoDeS parameter to this value at
    each iteration

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    point_function: callable
        A callable of two parameters: station, namespace, returning an iterator. Unrolling this iterator returns to
        us set values of the parameter
    """
    def __init__(self, parameter, point_function):
        self._parameter = parameter
        self._point_function = point_function
        super().__init__()

    def _setter_factory(self):
        for set_value in self._point_function(self._station, self._namespace):
            self._parameter.set(set_value)
            yield ParameterSweep.parameter_log_format(self._parameter, set_value)


class FunctionSweep(BaseSweepObject):
    """
    Sweep independent parameters by looping over set point values and calling a set function

    Parameters
    ----------
    set_function: callable
        A callable of three parameters: station, namespace, set_value. This returns a dictionary containing arbitrary
        information about the value set and any measurements performed (or it can contain any information that needs to
        be added in the final dataset
    point_function: callable
        A callable of two parameters: station, namespace, returning an iterator. Unrolling this iterator returns to
        us set values of the parameter
    """
    def __init__(self, set_function, point_function):
        self._set_function = set_function
        self._point_function = point_function
        super().__init__()

    def _setter_factory(self):
        for set_value in self._point_function(self._station, self._namespace):
            yield self._set_function(self._station, self._namespace, set_value)


class FunctionWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once and returns the result of a function

    Parameters
    ----------
    func: callable
        callable of station, namespace
    """
    def __init__(self, func):
        self._func = func
        super().__init__()

    def _setter_factory(self):
        yield self._func(self._station, self._namespace)


class ParameterWrapper(BaseSweepObject):
    """
    A wrapper class which iterates once ans returns the value of a QCoDeS parameter

    Parameters
    ----------
    parameter: qcodes.StandardParameter
    """
    def __init__(self, parameter):
        self._parameter = parameter
        super().__init__()

    def _setter_factory(self):
        value = self._parameter()
        yield BaseSweepObject.parameter_log_format(self._parameter, value, setting=False)


class HardwareSweep(BaseSweepObject):
    """
    Hardware sweep. In effect the sweeping of parameters is performed by the hardware and we read the measurement values
    of this sweep in one go by reading the instrument buffer. Therefore, the value key of the measurement dictionary is
    an array or list.

    Parameters
    ----------
    func: callable
        A callable of station, namespace. The return value of this function is a dictionary

    Example
    -------
    >>> hardware_measurement = lambda station, namespace: {"some_measurement": {"unit": "V", "value": [0, 1, 2, 3]}}
    >>> so = HardwareSweep(hardware_measurement)
    >>> for iteration in so:
    >>>     print(iteration)
    >>> {"some_measurement": {"unit": "V", "value": 0}
    >>> {"some_measurement": {"unit": "V", "value": 1}
    >>> {"some_measurement": {"unit": "V", "value": 2}
    >>> {"some_measurement": {"unit": "V", "value": 3}

    See docs/hardware_sweep_example.ipynb for further examples.
    """
    def __init__(self, func):
        self._func = func
        self._value_sizes = set()
        super().__init__()

    def _make_result_generator(self, param_result):
        value = copy.copy(param_result["value"])

        if not isinstance(value, collections.Sized):
            value = itertools.repeat(value)
            self._value_sizes.add(1)
        else:
            self._value_sizes.add(len(value))

        def inner():
            for v in value:
                # Make a fresh copy of param_result every time to avoid troubles with immutable objects.
                # See test_hardware_sweep_sanity in the sweep object test module.
                p = dict(param_result)
                p["value"] = v
                yield p

        return inner

    def _setter_factory(self):
        sweep_result = self._func(self._station, self._namespace)
        result_generators = []

        for param_result in sweep_result.values():
            result_generators.append(self._make_result_generator(param_result)())

        if len(self._value_sizes) > 2 or (len(self._value_sizes) == 2 and min(self._value_sizes) > 1):
            raise RuntimeError("The values returned by the hardware sweep should all be the "
                               "same size. Values of size 1 are allowed and these values will be repeated")

        for _ in range(max(self._value_sizes)):
            iteration_result = {k: next(v) for k, v in zip(sweep_result.keys(), result_generators)}
            yield iteration_result


class NestedSweep(CompoundSweep):
    """
    Nest multiple sweep objects. This is for example very useful when performing two or higher dimensional scans
    (e.g. sweep two gate voltages and measure a current at each coordinate (gate1, gate2).

    Notes
    -----
    We produce a nested sweep object of arbitrary depth by first defining a function which nests just two sweep objects

        product = two_product(so1, so2)

    A third order nest is then achieved like so:

        product = two_product(so1, two_product(so2, so3))

    A fourth order by

        product = two_product(so1, two_product(so2, two_product(so3, so4)))

    Etc...
    """

    @staticmethod
    def _two_product(sweep_object1, sweep_object2):
        """
        Parameters
        ----------
        sweep_object1: BaseSweepObject
        sweep_object2: BaseSweepObject

        Returns
        -------
        BaseSweepObject
        """
        def inner():
            for result2 in sweep_object2:
                for result1 in sweep_object1:
                    result1.update(result2)

                    yield result1

        return IteratorSweep(inner)

    def _setter_factory(self):
        prod = self._sweep_objects[0]
        for so in self._sweep_objects[1:]:
            prod = self._two_product(so, prod)

        return prod


class ZipSweep(CompoundSweep):
    """
    Zip multiple sweep objects. Unlike a nested sweep, we will produce a 1D sweep
    """
    @staticmethod
    def _combine_dictionaries(dictionaries):
        combined = {}
        for d in dictionaries:
            combined.update(d)
        return combined

    def _setter_factory(self):
        for results in zip(*self._sweep_objects):
            yield ZipSweep._combine_dictionaries(results)


class ChainSweep(CompoundSweep):
    """
    Chain a list of sweep object to run one after the other
    """
    def _setter_factory(self):
        for so in self._sweep_objects:
            for result in so:
                yield result


class TimeTrace(BaseSweepObject):
    """
    Make a "time sweep", that is, take a time trace

    Parameter
    ---------
    measure: callable
        callable of arguments  station, namespace, returning a dictionary with measurement results.
    delay: float
        The time in seconds between calling the measure function
    total_time: float
        The total duration of the time trace
    """
    def __init__(self, measure, delay, total_time):
        self._measure = measure
        self._delay = delay
        self._total_time = total_time
        super().__init__()

    def _setter_factory(self):
        t0 = time.time()
        t = t0
        while t - t0 < self._total_time:
            msg = self._measure(self._station, self._namespace)
            time_msg = {"time": {"unit": "s", "value": t, "independent_parameter": True}}
            msg.update(time_msg)
            yield msg
            time.sleep(self._delay)
            t = time.time()

# --------------  Sweep Factories ----------------------------------------------------------------------------


def sweep(obj, sweep_points):
    """
    A convenience function to create a 1D sweep object

    Parameters
    ----------
    obj: qcodes.StandardParameter or callable
        If callable, it shall be a callable of three parameters: station, namespace, set_value and shall return a
        dictionary
    sweep_points: iterable or callable returning a iterable
        If callable, it shall be a callable of two parameters: station, namespace and shall return an iterable

    Returns
    -------
    FunctionSweep or ParameterSweep
    """

    if not callable(sweep_points):
        point_function = lambda station, namespace: sweep_points
    else:
        point_function = sweep_points

    if not isinstance(obj, qcodes.Parameter):
        if not callable(obj):
            raise ValueError("The object to sweep over needs to either be a QCoDeS parameter or a function")

        return FunctionSweep(obj, point_function)
    else:
        return ParameterSweep(obj, point_function)


def nested_sweep(*sweep_objects):
    """
    A convenience function to make nested sweeps

    Parameters
    ----------
    sweep_objects: list, BaseSweepObject
    """
    return NestedSweep(sweep_objects)


def zip_sweep(*sweep_objects):
    """
    A convenience function to make a zipped sweep

    Parameters
    ----------
    sweep_objects: list, BaseSweepObject
    """
    return ZipSweep(sweep_objects)


def time_trace(measure, delay, total_time):
    return TimeTrace(measure, delay, total_time)


def hardware_sweep(measurement_function):
    return HardwareSweep(measurement_function)
