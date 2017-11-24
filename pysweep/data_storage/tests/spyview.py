from hypothesis import given, settings
from hypothesis.strategies import integers
import pytest

import tempfile
from pysweep.data_storage.spyview import SpyviewWriter, SpyviewMetaWriter


def read_lines(file):
    file.seek(0)
    return file.read().decode().split("\n")


def test_no_independent_params_exception():

    def output_write_function(output):
        pass

    def meta_write_function(output):
        pass

    meta_writer = SpyviewMetaWriter(meta_write_function)
    writer = SpyviewWriter(output_write_function, meta_writer, max_buffer_size=1)

    m, n = 3, 3
    x = list(range(m))
    y = [ix ** 2 + 3 for ix in x]

    x_dicts = [{"x": {"unit": "u", "value": v}} for v in x]  # This should be an independent parameter, but Isn't
    y_dicts = [{"y": {"unit": "v", "value": v}} for v in y]

    for dict_list in zip(x_dicts, y_dicts):

        merge = {}
        for d in dict_list:
            merge.update(d)

        with pytest.raises(RuntimeError) as e_info:
            writer.add(merge)


@given(integers(min_value=3, max_value=700), integers(min_value=1, max_value=500))
@settings(max_examples=30)
def test_1d(m, max_buffer_size):

    output_file = tempfile.TemporaryFile()
    meta_output_file = tempfile.TemporaryFile(mode="rb+")

    def output_write_function(output):
        output_file.seek(0)
        output_file.write(output.encode())

    def meta_write_function(output):
        meta_output_file.seek(0)
        meta_output_file.write(output.encode())

    meta_writer = SpyviewMetaWriter(meta_write_function)
    writer = SpyviewWriter(output_write_function, meta_writer, max_buffer_size=max_buffer_size)

    x = list(range(m))
    y = [ix ** 2 + 3 for ix in x]

    x_dicts = [{"x": {"unit": "u", "value": v, "independent_parameter": True}} for v in x]
    y_dicts = [{"y": {"unit": "v", "value": v}} for v in y]

    write_count = 0
    for dict_list in zip(x_dicts, y_dicts):

        merge = {}
        for d in dict_list:
            merge.update(d)

        writer.add(merge)
        write_count += 1
        non_empty_line_count = len([l for l in read_lines(output_file) if l != ""])
        assert non_empty_line_count == max_buffer_size * (write_count // max_buffer_size)

    writer.finalize()

    lines = read_lines(output_file)
    lines = (l for l in lines)

    for ix in range(m):
        assert next(lines) == "\t".join(map(str, [ix, 0, ix ** 2 + 3]))

    meta_debug_lines = read_lines(meta_output_file)
    compare = [str(i) for i in [m, min(x), max(x), "x [u]", 1, 0, 0, "empty []", 1, 0, 1, "none", 3, "y [v]"]]

    assert meta_debug_lines == compare


@given(integers(min_value=3, max_value=7), integers(min_value=3, max_value=7), integers(min_value=1, max_value=500))
@settings(max_examples=30)
def test(n, m, max_buffer_size):

    output_file = tempfile.TemporaryFile()
    meta_output_file = tempfile.TemporaryFile(mode="rb+")

    def output_write_function(output):
        output_file.seek(0)
        output_file.write(output.encode())

    def meta_write_function(output):
        meta_output_file.seek(0)
        meta_output_file.write(output.encode())

    meta_writer = SpyviewMetaWriter(meta_write_function)
    writer = SpyviewWriter(output_write_function, meta_writer, max_buffer_size=max_buffer_size)

    x = m * list(range(n))
    y = [i//n for i in range(m * n)]
    z = [ix**2 + iy for ix, iy in zip(x, y)]

    x_dicts = [{"x": {"unit": "u", "value": v, "independent_parameter": True}} for v in x]
    y_dicts = [{"y": {"unit": "v", "value": v, "independent_parameter": True}} for v in y]
    z_dicts = [{"z": {"unit": "w", "value": v}} for v in z]

    write_count = 0
    for dict_list in zip(x_dicts, y_dicts, z_dicts):

        merge = {}
        for d in dict_list:
            merge.update(d)

        writer.add(merge)
        write_count += 1
        non_empty_line_count = len([l for l in read_lines(output_file) if l != ""])

        assert non_empty_line_count == max_buffer_size * (write_count // max_buffer_size)

    writer.finalize()

    lines = read_lines(output_file)
    lines = (l for l in lines)

    for iy in range(m):
        if iy > 0:
            assert next(lines) == ""

        for ix in range(n):
            iz = ix**2 + iy
            test_line = "\t".join(map(str, [ix, iy, iz]))
            assert next(lines) == test_line

    meta_debug_lines = read_lines(meta_output_file)
    compare = [str(i) for i in [n, min(x), max(x), "x [u]", m, max(y), min(y), "y [v]", 1, 0, 1, "none", 3, "z [w]"]]

    assert meta_debug_lines == compare


@given(integers(min_value=3, max_value=7), integers(min_value=3, max_value=7), integers(min_value=1, max_value=500))
@settings(max_examples=30)
def test_delayed(n, m, max_buffer_size):

    output_file = tempfile.TemporaryFile()
    meta_output_file = tempfile.TemporaryFile(mode="rb+")

    def output_write_function(output):
        output_file.seek(0)
        output_file.write(output.encode())

    def meta_write_function(output):
        meta_output_file.seek(0)
        meta_output_file.write(output.encode())

    def get_y_dict():
        return {"y": {"unit": "v", "value": [], "independent_parameter": True}}

    meta_writer = SpyviewMetaWriter(meta_write_function)
    writer = SpyviewWriter(output_write_function, meta_writer, max_buffer_size=max_buffer_size, delayed_parameters="y")

    x = m * list(range(n))
    y = [i//n for i in range(m * n)]
    z = [ix**2 + iy for ix, iy in zip(x, y)]

    x_dicts = [{"x": {"unit": "u", "value": v, "independent_parameter": True}} for v in x]
    z_dicts = [{"z": {"unit": "w", "value": v}} for v in z]
    y_dict = get_y_dict()

    for count, (x_dict, iy, z_dict) in enumerate(zip(x_dicts, y, z_dicts)):

        merge = {}
        for d in [x_dict, z_dict]:
            merge.update(d)

        y_dict["y"]["value"].append(iy)
        if count != 0 and count % (n * m // 2) == 0:
            merge.update(y_dict)
            y_dict = get_y_dict()

        writer.add(merge)

    writer.add(y_dict)
    writer.finalize()

    lines = read_lines(output_file)
    lines = (l for l in lines)

    for iy in range(m):
        if iy > 0:
            assert next(lines) == ""

        for ix in range(n):
            iz = ix**2 + iy
            test_line = "\t".join(map(str, [ix, iy, iz]))
            assert next(lines) == test_line

    meta_debug_lines = read_lines(meta_output_file)
    compare = [str(i) for i in [n, min(x), max(x), "x [u]", m, max(y), min(y), "y [v]", 1, 0, 1, "none", 3, "z [w]"]]

    assert meta_debug_lines == compare


@given(integers(min_value=5, max_value=10), integers(min_value=10, max_value=70), integers(min_value=1, max_value=500))
@settings(max_examples=3)
def test_blocks(n, m, max_buffer_size):

    output_file = tempfile.TemporaryFile()
    meta_output_file = tempfile.TemporaryFile(mode="rb+")

    def output_write_function(output):
        output_file.seek(0)
        output_file.write(output.encode())

    def meta_write_function(output):
        meta_output_file.seek(0)
        meta_output_file.write(output.encode())

    meta_writer = SpyviewMetaWriter(meta_write_function)
    writer = SpyviewWriter(output_write_function, meta_writer, max_buffer_size=max_buffer_size)

    x = list(range(m))
    z = [list(range(i, i + n)) for i in x]

    x_dicts = [{"x": {"unit": "u", "value": v, "independent_parameter": True}} for v in x]
    z_dicts = [{"z": {"unit": "w", "value": v}} for v in z]

    for dict_list in zip(x_dicts, z_dicts):

        merge = {}
        for d in dict_list:
            merge.update(d)

        writer.add(merge)

    writer.finalize()

    lines = read_lines(output_file)
    lines_g = (l for l in lines)

    for outer_count, z_block in enumerate(z):
        ix = x[outer_count]

        if outer_count > 0:
            assert next(lines_g) == ""

        for inner_count, iz in enumerate(z_block):
            test_line = "\t".join(map(str, [inner_count, ix, iz]))
            ln = next(lines_g)
            assert ln == test_line
