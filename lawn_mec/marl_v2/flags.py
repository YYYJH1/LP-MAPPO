from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Flags:
    f1: bool = True
    f2: bool = False
    f3: bool = True
    f4: bool = True
    f5: bool = True
    f6: bool = True
    f7: bool = True
    f8: bool = True
    f9: bool = True
    f14a: bool = False
    f14b: bool = False
    f14c: bool = False
    f15: bool = False
    f16: bool = False
    f17: bool = False

    def __post_init__(self):
        if any(type(v) is not bool for v in asdict(self).values()):
            raise TypeError('F13 flags must be booleans')
        if self.f15 and not (self.f1 and self.f6):
            raise ValueError('f15 extends the F1 timing candidates and the F6 request token')
        if self.f16 and not self.f6:
            raise ValueError('f16 extends the F6 location candidates')
        if self.f17 and not (self.f1 and self.f6 and self.f15 and self.f16):
            raise ValueError('f17 requires f1, f6, f15 and f16')

    @classmethod
    def legacy(cls):
        return cls(**{f'f{i}': False for i in range(1, 10)})


def flags_or_default(flags):
    return flags if isinstance(flags, Flags) else Flags(**(flags or {}))


RECORDED_WHEN_SET = ('f15', 'f16', 'f17')


def flags_record(flags):
    record = asdict(flags_or_default(flags))
    for name in RECORDED_WHEN_SET:
        if record[name] == Flags.__dataclass_fields__[name].default:
            del record[name]
    return record
