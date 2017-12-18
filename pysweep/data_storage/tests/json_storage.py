import pytest

from pysweep.data_storage.json_storage import JSONStorage


def test():
    a_values = range(5)
    b_values = [i+2 for i in a_values]

    a_dicts = [{"A": {"unit": "u1", "value": v}} for v in a_values]
    b_dicts = [{"B": {"unit": "u2", "value": v}} for v in b_values]

    storage = JSONStorage(unit="replace", value="append", independent_parameter="replace")
    for a, b in zip(a_dicts, b_dicts):
        merged = dict(a)
        merged.update(b)
        storage.add(merged)

    assert storage.output("A") == {
        "A": {"unit": "u1", "value": list(a_values)},
        "B": {"unit": "u2", "value": list(b_values)}
    }

    assert sorted(storage.keys()) == sorted(["A", "B"])

    with pytest.raises(ValueError):
        storage.output("C")


def test2():
    a_values = range(5)
    b_values = [i+2 for i in a_values]

    a_dicts = [{"A": {"unit": "u1", "value": v}} for v in a_values]
    b_dicts = [{"B": {"unit": "u2", "value": v}} for v in b_values]

    storage = JSONStorage(unit="replace", value="append", independent_parameter="replace")
    for a, b in zip(a_dicts, b_dicts):
        storage.add(a)
        storage.add(b)

    assert storage.output("A") == {"A": {"unit": "u1", "value": list(a_values)}}
    assert storage.output("B") == {"B": {"unit": "u2", "value": list(b_values)}}
    assert sorted(storage.keys()) == sorted(["A", "B"])

    with pytest.raises(ValueError):
        storage.output("C")


def test_list_values():
    """
    We have seen in the past a bug whereby if the measurement function returns an arrayed value and we call the output
    method of the JSON storage class twice, then the internal buffer of the storage gets corrupted. Test that this
    bug solved.
    """

    m, n = 4, 5
    values = [list(range(i, i + m)) for i in range(n)]
    dicts = [{"A": {"unit": "u", "value": v}} for v in values]

    storage = JSONStorage(unit="replace", value="append", independent_parameter="replace")
    for d in dicts:
        storage.add(d)

    o = storage.output("A")
    assert len(o["A"]["value"]) == m * n

    o = storage.output("A")
    assert len(o["A"]["value"]) == m * n
