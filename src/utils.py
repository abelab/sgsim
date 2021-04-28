import math
from collections.abc import Iterable
import sg


class Stat:
    AVERAGE = 0
    MIN = 1
    MAX = 2
    SUM = 3
    ITEMS = 4

    def __init__(self):
        self.stats: [float] = []

    def add(self, data: float) -> None:
        self.stats.append(data)

    def average(self) -> float:
        return sum(self.stats) / len(self.stats) if len(self.stats) > 0 else math.nan

    def min(self) -> float:
        return min(self.stats) if len(self.stats) > 0 else math.nan

    def max(self) -> float:
        return max(self.stats) if len(self.stats) > 0 else math.nan

    def sum(self) -> float:
        return sum(self.stats)

    def items(self) -> int:
        return len(self.stats)

    def percentile(self, percentile: float) -> float:
        if len(self.stats) == 0:
            return math.nan
        self.stats.sort()
        n = len(self.stats)
        i = math.floor((n-1) * percentile)
        reminder = n * percentile - i
        if i < n - 1:
            rc = self.stats[i] + (self.stats[i+1]-self.stats[i]) * reminder
        else:
            rc = self.stats[i]
        return rc

    def major_stats(self) -> [float]:
        """ returns a list, where list[Stat.AVERAGE] contains the average, list[Stat.MIN] contains the min, etc."""
        return [self.average(), self.min(), self.max(), self.sum(), self.items()]


class StatArray:
    def __init__(self):
        self.stats = {}

    def add(self, data: [float]) -> None:
        for i, d in enumerate(data):
            if self.stats.get(i) is None:
                self.stats[i] = Stat()
            stat: Stat = self.stats[i]
            stat.add(d)

    def average(self) -> [float]:
        if len(self.stats) == 0:
            return []
        max_index = max(list(self.stats))
        results = [0.0] * (max_index + 1)
        for i, s in self.stats.items():
            results[i] = s.average()
        return results

    def get_stat(self, index: int) -> Stat:
        return self.stats[index]

    def get_stats(self, *indexes) -> [Stat]:
        rc = []
        for i in indexes:
            rc.append(self.stats[i])
        return rc


def list_str(itr: Iterable) -> str:
    return str([str(n) for n in itr])


def verbose(*args):
    if sg.VERBOSE:
        print(*args)


if __name__ == "__main__":
    a = StatArray()
    # a.add([1, 2, 3])
    # a.add([4, 5, 6, 7])
    print(a.average())
