import qcodes

from pysweep.parameter_alias import Alias, LinearMapping


class Powersource(qcodes.Instrument):
    def __init__(self, name):
        super().__init__(name)
        self._voltage = 0

        self.add_parameter(
            name="voltage",
            set_cmd=self._set_voltage,
            get_cmd=self._get_voltage,
            unit="mV"  # Stupid thing works in mV instead of Volt
        )

    def _set_voltage(self, value):
        self._voltage = value

    def _get_voltage(self):
        return self._voltage


def test():
    pwr = Powersource("pwr")

    mapping = LinearMapping("milli to unit")
    assert mapping.conversion_factor == 1E-3

    gate = Alias(pwr.voltage, name="gate", mapping=mapping, label="gate voltage", unit="V")
    value_to_set = 3
    gate(value_to_set)

    # Test that the values have been correctly set
    assert gate() == value_to_set
    assert pwr.voltage() == value_to_set / mapping.conversion_factor

    # Test that the snap shot reflects this
    instrument_snapshot = pwr.snapshot()
    assert "gate" in instrument_snapshot["parameters"]
    assert instrument_snapshot["parameters"]["gate"]["value"] == value_to_set
    assert instrument_snapshot["parameters"]["voltage"]["value"] == value_to_set / mapping.conversion_factor

    # Test that we have useful meta data
    alias_metadata = gate.metadata
    assert alias_metadata["alias_of"] == "{}.{}".format(pwr.name, pwr.voltage.name)
    assert alias_metadata["mapping"] == str(mapping)


def test_mappings():
    assert LinearMapping("milli to unit").conversion_factor == 1E-3 / 1
    assert LinearMapping("unit to milli").conversion_factor == 1 / 1E-3
    assert LinearMapping("mega to nano").conversion_factor == 1E6 / 1E-9
    assert LinearMapping("kilo to giga").conversion_factor == 1E3 / 1E9

    assert LinearMapping("milli to unit")(1) == 1E3
    assert LinearMapping("unit to milli")(1) == 1E-3
    assert LinearMapping("mega to nano")(1) == 1E-9 / 1E6
    assert LinearMapping("kilo to giga")(1) == 1E9 / 1E3

    assert LinearMapping("milli to unit").inverse(1) == 1E-3
    assert LinearMapping("unit to milli").inverse(1) == 1E3
    assert LinearMapping("mega to nano").inverse(1) == 1E6 / 1E-9
    assert LinearMapping("kilo to giga").inverse(1) == 1E3 / 1E9


def test_free_floating_param():

    class GetterSetter:
        def __init__(self):
            self.value = None

        def setter(self, value):
            self.value = value

        def getter(self):
            return self.value

    gs = GetterSetter()

    param = qcodes.StandardParameter(
        "param",
        unit="-",
        set_cmd=gs.setter,
        get_cmd=gs.getter
    )

    conversion_factor = 10
    alias = Alias(param, "alias", mapping=conversion_factor)  # Note that we can also give a number

    value_to_set = 2
    alias(value_to_set)

    assert alias() == value_to_set
    assert param() == value_to_set / conversion_factor

