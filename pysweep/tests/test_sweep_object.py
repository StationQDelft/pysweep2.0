import numpy as np

from pysweep.sweep_object import sweep, nested_sweep, ParameterSweep, ChainSweep, HardwareSweep
from pysweep.tests.testing_utilities import equivalence_test, run_test_function

from .testing_utilities import sorted_dict

param_log_format = ParameterSweep.parameter_log_format


def test_sanity():

    # Test that this ...
    def test(params, values, stdio, measure, namespace):
        for i in sweep(params[0], values[0]):
            stdio.write(sorted_dict(i))

        return str(stdio)

    # Is equivalent to this
    def compare(params, values, stdio, measure, namespace):
        for value in values[0]:
            params[0].set(value)
            dct = sorted_dict(param_log_format(params[0], value))
            stdio.write(dct)

        return str(stdio)

    equivalence_test(test, compare)


def test_product():

    def test(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]

        so = nested_sweep(
            sweep(param1, sweep_values1),
            sweep(param2, sweep_values2),
            sweep(param3, sweep_values3),
            sweep(param4, sweep_values4),
        )

        for i in so:
            stdio.write(sorted_dict(i))

        return str(stdio)

    def compare(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]

        for value1 in sweep_values1:
            param1.set(value1)
            for value2 in sweep_values2:
                param2.set(value2)
                for value3 in sweep_values3:
                    param3.set(value3)
                    for value4 in sweep_values4:
                        param4.set(value4)

                        values = [value1, value2, value3, value4]
                        params = [param1, param2, param3, param4]

                        dct = sorted_dict([
                            param_log_format(p, value) for p, value in zip(params, values)])

                        stdio.write(dct)

        return str(stdio)

    equivalence_test(test, compare)


def test_measure():

    def test1(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure = measures[0]

        so = nested_sweep(
            sweep(param1, sweep_values1),
            sweep(param2, sweep_values2),
            measure,
            sweep(param3, sweep_values3),
            sweep(param4, sweep_values4),
        )

        for i in so.set_namespace(namespace):
            stdio.write(sorted_dict(i))

        return str(stdio)

    def test2(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure = measures[0]

        so = ChainSweep([(
            sweep(param1, sweep_values1),
            sweep(param2, sweep_values2),
            measure,
            sweep(param3, sweep_values3),
            sweep(param4, sweep_values4),
        )])

        for i in so.set_namespace(namespace):
            stdio.write(sorted_dict(i))

        return str(stdio)

    def compare(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure = measures[0]

        for value1 in sweep_values1:
            param1.set(value1)
            for value2 in sweep_values2:
                param2.set(value2)
                measure_dct = measure(None, namespace)
                for value3 in sweep_values3:
                    param3.set(value3)
                    for value4 in sweep_values4:
                        param4.set(value4)

                        values = [value1, value2, value3, value4]
                        params = [param1, param2, param3, param4]

                        dct = sorted_dict([
                            param_log_format(p, value) for p, value in zip(params, values)])
                        measure_dct.update(dct)

                        stdio.write(measure_dct)

        return str(stdio)

    equivalence_test(test1, compare)
    equivalence_test(test2, compare)


def test_chain_nest():

    def test1(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure1, measure2 = measures[:2]

        so = ChainSweep([(
            sweep(param1, sweep_values1),
            sweep(param2, sweep_values2),
            measure1,
        ), (
            sweep(param3, sweep_values3),
            sweep(param4, sweep_values4),
            measure2
        )])

        for i in so.set_namespace(namespace):
            stdio.write(sorted_dict(i))

        return str(stdio)

    def test2(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure1, measure2 = measures[:2]

        so1 = \
            sweep(param1, sweep_values1)(
                sweep(param2, sweep_values2)(
                    measure1
                )
            )

        so2 = \
            sweep(param3, sweep_values3)(
                sweep(param4, sweep_values4)(
                    measure2
                )
            )

        so = ChainSweep([so1, so2])

        for i in so.set_namespace(namespace):
            stdio.write(sorted_dict(i))

        return str(stdio)

    def compare(params, values, stdio, measures, namespace):

        param1, param2, param3, param4 = params[:4]
        sweep_values1, sweep_values2, sweep_values3, sweep_values4 = values[:4]
        measure1, measure2 = measures[:2]

        for value1 in sweep_values1:
            param1(value1)
            for value2 in sweep_values2:
                param2(value2)
                measure_dct = measure1(None, namespace)

                values = [value1, value2]
                params = [param1, param2]

                dct = sorted_dict([
                    param_log_format(p, value) for p, value in zip(params, values)])
                measure_dct.update(dct)

                stdio.write(measure_dct)

        for value3 in sweep_values3:
            param3(value3)
            for value4 in sweep_values4:
                param4(value4)
                measure_dct = measure2(None, namespace)

                values = [value3, value4]
                params = [param3, param4]

                dct = sorted_dict([
                    param_log_format(p, value) for p, value in zip(params, values)])
                measure_dct.update(dct)

                stdio.write(measure_dct)

        return str(stdio)

    equivalence_test(test1, compare)
    equivalence_test(test2, compare)


def test_hardware_sweep():

    def hardware_measurement(values):
        def inner(station, namespace):
            v = namespace.v
            namespace.v += 1
            read_values = [i * v for i in values]
            return {
                "measurement": {"unit": "V", "value": read_values},
                "index": {"unit": "-", "value": list(range(len(read_values))), "independent_parameter": True},
                "other": {"unit": "T", "value": values[0] - v}
            }

        return inner

    def test(params, values, stdio, measure, namespace):

        namespace.v = 0
        measurement_function = hardware_measurement(values[0])
        p = params[0]
        value = values[1]

        so = sweep(p, value)
        hwso = HardwareSweep(measurement_function)

        for i in so(hwso).set_namespace(namespace):
            stdio.write(i)

        return str(stdio)

    def compare(params, values, stdio, measure, namespace):

        p = params[0]
        v = 0

        for value1 in values[1]:
            p(value1)
            for count, value0 in enumerate(values[0]):
                i = {
                    "measurement": {"unit": "V", "value": v * value0},
                    "index": {"unit": "-", "value": count, "independent_parameter": True},
                    "other": {"unit": "T", "value": values[0][0] - v}
                }
                i.update(param_log_format(p, value1))
                stdio.write(i)
            v += 1

        return str(stdio)

    equivalence_test(test, compare)
