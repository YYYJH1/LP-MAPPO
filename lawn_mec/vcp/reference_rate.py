REFERENCE_RATES = ('nominal', 'certified')
DEFAULT_REFERENCE_RATE = 'nominal'


def check_reference_rate(rate):
    if rate not in REFERENCE_RATES:
        raise ValueError(f'reference rate must be one of {REFERENCE_RATES}, got {rate!r}')
    return rate
