
def is_ordered_inclusive(a, b, c) -> bool:
    if a <= b <= c:
        return True
    if b <= c <= a:
        return True
    if c <= a <= b:
        return True
    return False


def is_ordered(start: int, start_inclusive: bool, val: int, end: int, end_inclusive: bool) -> bool:
    if start == end:
        return start_inclusive != end_inclusive or start == val
    rc = is_ordered_inclusive(start, val, end)
    if rc:
        if start == val:
            rc = start_inclusive
    if rc:
        if val == end:
            rc = end_inclusive
    return rc
