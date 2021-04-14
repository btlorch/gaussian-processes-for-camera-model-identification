def nullable_int(val):
    if not val:
        return None

    return int(val)


def nullable_float(val):
    if not val:
        return None

    return float(val)
