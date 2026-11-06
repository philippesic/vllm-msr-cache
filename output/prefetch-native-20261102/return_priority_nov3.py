"""One gateway clock; fixed credit for released continuation requests."""
def gateway_priority(submitted_ns, origin_ns, turn, credit_ms):
    for name, value in (("submitted_ns", submitted_ns), ("origin_ns", origin_ns), ("turn", turn), ("credit_ms", credit_ms)):
        if type(value) is not int:
            raise TypeError(name + " must be an integer")
    if submitted_ns < origin_ns or origin_ns < 0 or turn < 0 or credit_ms not in (0, 2000, 8000):
        raise ValueError("Invalid gateway clock, turn, or fixed priority credit")
    if credit_ms == 0:
        return 0
    return (submitted_ns - origin_ns) // 1_000_000 - (credit_ms if turn > 0 else 0)
