from qcodes import Station

from .testing_utilities import equivalence_test, run_test_function

from pysweep.measurement import Measurement
from pysweep import sweep

Measurement.use_storage("json")
station = Station()
Measurement.set_station(station)


def test_sanity():

    def test(params, values, stdio, measure, namespace):

        p = params[0]
        setup, cleanup = measure[:2]
        sweep_values = values[0]

        measurement = Measurement(
            setup,
            cleanup,
            sweep(p, sweep_values)
        ).run()

        out = measurement.output(p.label)
        assert out == {p.label: {"unit": p.unit, "value": list(sweep_values), "independent_parameter": True}}

        return str(stdio)

    def compare(params, values, stdio, measure, namespace):

        p = params[0]
        setup, cleanup = measure[:2]
        sweep_values = values[0]

        setup(None, namespace)
        for value in sweep_values:
            p(value)
        cleanup(None, namespace)

        return str(stdio)

    equivalence_test(test, compare)


def test_simple_measure():

    def test(params, values, stdio, measure, namespace):

        p = params[0]
        setup, cleanup, measure_main = measure[:3]
        sweep_values = values[0]

        measurement = Measurement(
            setup,
            cleanup,
            (
                sweep(p, sweep_values),
                measure_main
            )
        ).run()

        out = measurement.output(p.label)
        assert out[p.label] == {"unit": p.unit, "value": list(sweep_values), "independent_parameter": True}
        assert measure_main.name in out

        return str(stdio)

    def compare(params, values, stdio, measure, namespace):

        p = params[0]
        setup, cleanup, measure_main = measure[:3]
        sweep_values = values[0]

        setup(None, namespace)
        for value in sweep_values:
            p(value)
            measure_main(None, namespace)
        cleanup(None, namespace)

        return str(stdio)

    equivalence_test(test, compare)


def test_simple_nested():

    def test1(params, values, stdio, measure, namespace):

        p1, p2 = params[:2]
        setup, cleanup, measure_main = measure[:3]
        sweep_values = values[0]

        measurement = Measurement(
            setup,
            cleanup,
            (
                sweep(p1, sweep_values),
                measure_main,
                p2
            )
        ).run()

        out = measurement.output(p1.label)
        assert out[p1.label] == {"unit": p1.unit, "value": list(sweep_values), "independent_parameter": True}
        assert out[p2.label] == {"unit": p2.unit, "value": len(sweep_values) * [0]}
        assert measure_main.name in out

        return str(stdio)

    def test2(params, values, stdio, measure, namespace):

        p1, p2 = params[:2]
        setup, cleanup, measure_main = measure[:3]
        sweep_values = values[0]

        measurement = Measurement(
            setup,
            cleanup,
            (
                sweep(p1, sweep_values),
                [
                    measure_main,
                    p2
                ]
            )
        ).run()

        out1 = measurement.output(p2.label)
        assert out1[p1.label] == {"unit": p1.unit, "value": list(sweep_values), "independent_parameter": True}
        assert out1[p2.label] == {"unit": p2.unit, "value": len(sweep_values) * [0]}
        assert measure_main.name not in out1

        out2 = measurement.output(measure_main.name)
        assert out2[p1.label] == {"unit": p1.unit, "value": list(sweep_values), "independent_parameter": True}
        assert measure_main.name in out2

        return str(stdio)

    def compare(params, values, stdio, measure, namespace):

        p1, p2 = params[:2]
        setup, cleanup, measure_main = measure[:3]
        sweep_values = values[0]

        setup(None, namespace)
        for value in sweep_values:
            p1(value)
            measure_main(None, namespace)
            p2()
        cleanup(None, namespace)

        return str(stdio)

    equivalence_test(test1, compare)
    equivalence_test(test2, compare)

