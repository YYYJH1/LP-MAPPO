from collections import deque
from dataclasses import dataclass, field
from importlib import import_module
from typing import Any, Protocol

DESIGN_OPTIONS = tuple(dict.fromkeys("ABCDEFGHLPQRSGXYZ-"))


@dataclass(frozen=True)
class FieldSpec:
    name: str
    options: tuple[str, ...]
    forced: bool

    def __post_init__(self):
        if not self.name or not isinstance(self.options, tuple) or not self.options:
            raise ValueError("A field needs a name and a nonempty tuple of options")
        if any(not isinstance(o, str) or not o for o in self.options):
            raise ValueError("Options must be nonempty strings")
        if len(set(self.options)) != len(self.options):
            raise ValueError("Duplicate field options")
        if self.forced and len(self.options) != 1:
            raise ValueError("A forced field must have exactly one option")


class FieldGrammar(Protocol):
    def reset(self, instance_key: str) -> None: ...
    def next_field(self, prefix: tuple[str, ...]) -> FieldSpec | None: ...
    def render_prefix_text(self, prefix: tuple[str, ...]) -> str: ...


@dataclass(frozen=True)
class GrammarSpec:
    factory: str = "lawn_mec.llm.grammar:ToyGrammar"
    kwargs: dict[str, Any] = field(default_factory=dict)

    def build(self, instance_key: str) -> FieldGrammar:
        module, name = self.factory.split(":")
        constructor = getattr(import_module(module), name)
        grammar = constructor(**self.kwargs)
        grammar.reset(instance_key)
        return grammar


class ToyGrammar:
    options = DESIGN_OPTIONS
    fields_per_block = 4

    def __init__(self, blocks: int = 2):
        if blocks < 1:
            raise ValueError("blocks must be positive")
        self.blocks = blocks
        self.instance_key = ""

    def reset(self, instance_key: str) -> None:
        self.instance_key = instance_key

    def next_field(self, prefix: tuple[str, ...]) -> FieldSpec | None:
        u, offset = divmod(len(prefix), self.fields_per_block)
        if u >= self.blocks:
            if u == self.blocks and offset == 0:
                return None
            raise ValueError("Prefix extends beyond grammar")
        if offset == 0:
            options = ("S",) if u and prefix[(u - 1) * 4] == "G" else ("S", "G")
        elif offset == 1:
            options = (("-",) if prefix[u * 4] == "G" else
                       (("P", "Q", "R") if self.instance_key == "remote-only"
                        else ("L", "P", "Q", "R")))
        elif offset == 2:
            options = ("-",) if prefix[-1] in ("-", "L") else ("X", "Y", "Z")
        else:
            options = ("-",)
        return FieldSpec(f"u{u}.{'g mu priority end'.split()[offset]}",
                         options, len(options) == 1)

    def render_prefix_text(self, prefix: tuple[str, ...]) -> str:
        spec = self.next_field(prefix)
        return "" if spec is None else f"\n{spec.name}="


@dataclass(frozen=True)
class CanonicalOptions:
    token_ids: dict[str, int]
    aliases: dict[str, str]

    @classmethod
    def build(cls, tokenizer, options) -> "CanonicalOptions":
        options = tuple(sorted(set(options)))
        token_ids, aliases = {}, {}
        special = set(tokenizer.all_special_ids)

        def single(text):
            ids = tokenizer.encode(text, add_special_tokens=False)
            return ids[0] if len(ids) == 1 and ids[0] not in special else None

        for option in options:
            token = single(option)
            if token is not None and token not in token_ids.values():
                token_ids[option], aliases[option] = token, option
        candidates = [p + c for p in (" ", "", "_")
                      for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789"]
        for option in options:
            if option in token_ids:
                continue
            for alias in candidates:
                token = single(alias)
                if token is not None and token not in token_ids.values():
                    token_ids[option], aliases[option] = token, alias
                    break
            else:
                raise ValueError(f"No unique single-token alias for {option!r}")
        result = cls(token_ids, aliases)
        result.validate(tokenizer)
        return result

    def validate(self, tokenizer):
        if not self.token_ids or self.token_ids.keys() != self.aliases.keys():
            raise ValueError("Incomplete canonical mapping")
        if len(set(self.token_ids.values())) != len(self.token_ids):
            raise ValueError("Canonical IDs must be globally injective")
        for option, token in self.token_ids.items():
            if tokenizer.encode(self.aliases[option], add_special_tokens=False) != [token]:
                raise ValueError(f"Alias is not one canonical token: {option}")
            if token in tokenizer.all_special_ids:
                raise ValueError("Special tokens cannot represent options")

    def ids(self, spec: FieldSpec) -> tuple[int, ...]:
        return tuple(self.token_ids[o] for o in spec.options)


@dataclass(frozen=True)
class FieldTrace:
    name: str
    position: int
    allowed_ids: tuple[int, ...]
    selected: int
    forced: bool


@dataclass(frozen=True)
class SequenceTrace:
    instance_key: str
    prompt: str
    prefix: tuple[str, ...]
    token_ids: tuple[int, ...]
    fields: tuple[FieldTrace, ...]


def encode_trace(tokenizer, mapping: CanonicalOptions, spec: GrammarSpec,
                 instance_key: str, prompt: str, prefix: tuple[str, ...]) -> SequenceTrace:
    grammar = spec.build(instance_key)
    ids = list(tokenizer.encode(prompt, add_special_tokens=False))
    if not ids:
        raise ValueError("Prompt must contain at least one token")
    fields = []
    for i, option in enumerate(prefix):
        current = grammar.next_field(prefix[:i])
        if current is None or option not in current.options:
            raise ValueError(f"Illegal option {option!r} at field {i}")
        ids.extend(tokenizer.encode(grammar.render_prefix_text(prefix[:i]),
                                    add_special_tokens=False))
        allowed = mapping.ids(current)
        fields.append(FieldTrace(current.name, len(ids), allowed,
                                 current.options.index(option), current.forced))
        ids.append(mapping.token_ids[option])
    return SequenceTrace(instance_key, prompt, prefix, tuple(ids), tuple(fields))


class TokenMachine:
    def __init__(self, tokenizer, mapping: CanonicalOptions, spec: GrammarSpec,
                 instance_key: str, prefix=(), stop_after_fields: int | None = None):
        self.tokenizer, self.mapping = tokenizer, mapping
        self.grammar = spec.build(instance_key)
        self.prefix = tuple(prefix)
        self.stop_after_fields = stop_after_fields
        if stop_after_fields is not None and stop_after_fields < len(prefix):
            raise ValueError("Block boundary precedes prefix")
        for i, option in enumerate(self.prefix):
            f = self.grammar.next_field(self.prefix[:i])
            if f is None or option not in f.options:
                raise ValueError("Invalid resumed prefix")
        self.ended = False
        self.consumed = 0
        self._prepare()

    def _prepare(self):
        self.field = self.grammar.next_field(self.prefix)
        if self.stop_after_fields == len(self.prefix):
            self.field = None
        self.template = deque(self.tokenizer.encode(
            self.grammar.render_prefix_text(self.prefix), add_special_tokens=False)
            if self.field is not None else [])

    @property
    def at_field(self):
        return not self.template and self.field is not None

    def allowed_ids(self):
        if self.ended:
            raise ValueError("Tokens requested after EOS")
        if self.template:
            return (self.template[0],)
        if self.field is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("Tokenizer must define EOS")
            return (self.tokenizer.eos_token_id,)
        return self.mapping.ids(self.field)

    def consume(self, token: int):
        allowed = self.allowed_ids()
        if token not in allowed:
            raise ValueError(f"Token {token} outside allowed set {allowed}")
        if self.template:
            self.template.popleft()
        elif self.field is None:
            self.ended = True
        else:
            self.prefix += (self.field.options[allowed.index(token)],)
            self._prepare()
        self.consumed += 1

    def sync(self, output_ids):
        if len(output_ids) < self.consumed:
            raise ValueError("Output history shrank; recreate state on rescheduling")
        for token in output_ids[self.consumed:]:
            self.consume(token)
