import time
from collections import defaultdict


def sleep(t):
    def inner(station, namespace):
        time.sleep(t)
        return {}
    return inner


class DictMerge:
    def __init__(self, **strategy):
        self._strategy = DictMerge._add_default(strategy)

    @staticmethod
    def _add_default(dictionary):
        result = defaultdict(lambda: "dict_merge")
        for k, v in dictionary.items():
            result[k] = v

        return result

    @staticmethod
    def _make_list(v):
        if isinstance(v, list):
            return list(v)  # Return a copy!
        return list([v])

    def _merge_two(self, d1, d2):
        result = dict(d1)
        for k, v in d2.items():
            if k in result:

                strategy = self._strategy[k]

                m = None
                if strategy == "append":
                    m = self._make_list(result[k])
                    m.extend(self._make_list(v))
                elif strategy == "replace":
                    m = v
                elif strategy == "dict_merge":
                    m = self._merge_two(v, result[k])

                result[k] = m
            else:
                result[k] = v

        return result

    def merge(self, dicts):
        result = dict(dicts[0])
        for d in dicts[1:]:
            result = self._merge_two(result, dict(d))
        return result


