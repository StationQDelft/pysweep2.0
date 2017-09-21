# PySweep 2.0 interface and design decisions
## Introduction
PySweep is a framework intended to be used on top of qcodes [QCoDeS](https://github.com/QCoDeS/Qcodes) in order to define measurements flexibly. At the most general level, a measurement has dependent and independant variables with setup and clean up methods. These need to be specified somehow. We propose the following structure:  

```python 
 measurement_table = {
  "independent_variables": {
    "gate1": {
      "unit": "V",
      "set_function": some.instrument.set, 
      "values": iterable_values
     }
     "gate2": {
        "unit": "V",
        "set_function": other.instrument.set, 
        "values": generator_values
     }
  },
  "dependent_variables": {
    "source_drain": {
    "unit": "A", 
    "get_function": yet_another.instrument.get
    }
  }
}
```

In the above example we have defined two independent variables. Our measurement loop therefore will be a nested loop where the first independent variable will be located in the inner most loop and will be the one which is sweeping most frequently.  

We can also couple to independent variables together to get a "co-sweep" like so: 

```python
"gate1, gate2" : {
 "unit": "V, V"
 "set_function": (some.instrument.set, other.instrument.set),
 "values": (iterable_values1, iterable_values2)
}
```

The measurement table will also ensure the proper formatting and labeling of the resulting measurement file. We note that the "values" field of indepenent variables can be iterators or generators. This will allow us to introduce considerable flexibility (for example, introduce feedback and adaptive stepping in calibration measurements). 

## Measurement setup and cleanup 

We however also need some way of defining the measurement setup and cleanup. The setup brings the hardware in a state making it ready to perform a measurement. This could for example be instructing a lock-in amplifier to respond to triggers when these are send or putting an oscilloscope in the correct measruement ranges. A cleanup ensure that the instruments are left in defined setting after the measurement has concluded. To enable all of this, we propose the following class definition: 

```python
class MyMeasurement(pysweep.BaseMeasurement):
    def setup(self):
        some.instrument.set(0)

    def measure(self, namespace):
        measurement_table = {
            "independent_variables": {
                "gate1": {
                    "unit": "V",
                    "set_function": some.instrument.set,
                    "values": iterable_values
                },
                "gate2": {
                    "unit": "V",
                    "set_function": other.instrument.set,
                    "values": generator_values
                }
            },
            "dependent_variables": {
                "source_drain": {
                    "unit": "A",
                    "get_function": yet_another.instrument.get
                }
            }
        }

        return measurement_table

    def cleanup(self, namespace):
        some.instrument.set(0)
```

## Hardware Triggering

Consider a measurement where we sweep a gate voltage over consecutive values and at each voltage we send a trigger signal to a measurement device to measure a source drain current. The measurement device will use an internal buffer to store the measured value. We want to read out the internal buffer in one go when either we are at the end of our measurement, or when the buffer is full. We propose the following schema to define such a measurement: 

```python
class MyMeasurement(pysweep.BaseMeasurement):
    def setup(self):
        some.instrument.set(0)

    def measure(self, namespace):
        measurement_table = {
            "independent_variables": {
                "gate1": {
                    "unit": "V",
                    "set_function": some.instrument.set,
                    "values": iterable_values, 
                    "at_each": {
                        "function": measurement_instrument.trigger
                    }
                },
                "gate2": {
                    "unit": "V",
                    "set_function": other.instrument.set,
                    "values": generator_values, 
                    "at_end": {
                        "function": measurement_instrument.force_read
                        "args": (True,)
                    }
                }
            },
            "dependent_variables": {
                "source_drain": {
                    "unit": "A",
                    "get_function": measurement_instrument.read_buffer
                }
            }
        }

        return measurement_table

    def cleanup(self, namespace):
        some.instrument.set(0)
```
In the inner loop we send a trigger for each voltage we set on gate 1. We then want to aqcuire the dependent variables and call read buffer. This function should return the string "delayed_N" until either a "force_read" flag becomes true, or if the number of triggers recieved is equal to the buffer size. Here N is any positive integer. When a buffer is read out, the values returned will have the following format: 

```python
{
    "delayed_0": 2.3E-4,
    "delayed_1": 2.2E-4
    ...
}
```
The delayed values will retroactively be filled in 
