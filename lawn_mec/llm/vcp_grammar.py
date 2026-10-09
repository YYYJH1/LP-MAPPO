import contextlib
import copyreg
from functools import cached_property, lru_cache
import gc
import hashlib
import json
import os
from pathlib import Path
import pickle
import sys
import types

from .grammar import FieldSpec
from lawn_mec.vcp.reference_rate import DEFAULT_REFERENCE_RATE, check_reference_rate

_VCP = Path(__file__).resolve().parents[1] / "vcp"
GRAMMAR_VERSION = "vcp-fields-v1-" + hashlib.sha256(
    (_VCP / "grammar.py").read_bytes()).hexdigest()[:16]


def _identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return str(path), stat.st_mtime_ns, stat.st_size


@lru_cache(maxsize=16)
def _registry(identity):
    path = Path(identity[0])
    entries = json.loads(path.read_text())
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Registry must be a nonempty key -> instance JSON path map")
    if any(not isinstance(k, str) or not isinstance(v, str) for k, v in entries.items()):
        raise ValueError("Registry keys and paths must be strings")
    return {key: str((path.parent / value).resolve()) for key, value in entries.items()}


def load_registry(registry):
    return dict(_registry(_identity(registry)))


_REFERENCE_RATE = DEFAULT_REFERENCE_RATE


def set_reference_rate(rate):
    global _REFERENCE_RATE
    _REFERENCE_RATE = check_reference_rate(rate)


def current_reference_rate():
    return _REFERENCE_RATE


def resolve_reference_rate(rate=None):
    return _REFERENCE_RATE if rate is None else check_reference_rate(rate)


class InstanceContext:
    reference_rate = DEFAULT_REFERENCE_RATE

    def __init__(self, identity, reference_rate=DEFAULT_REFERENCE_RATE):
        from lawn_mec.vcp.tables import CommitmentTables

        self.path = identity[0]
        raw = Path(self.path).read_bytes()
        self.sha256 = hashlib.sha256(raw).hexdigest()
        self.instance = json.loads(raw)
        self.tables = CommitmentTables(self.instance, reference_rate=reference_rate)
        if reference_rate != DEFAULT_REFERENCE_RATE:
            self.reference_rate = reference_rate

    @cached_property
    def dp(self):
        from lawn_mec.vcp.dp_commit import RelaxedDP

        return RelaxedDP(self.tables)

    @cached_property
    def reference(self):
        return self.dp.commitment()


CONTEXT_CACHE_ENV = "LAWN_CONTEXT_CACHE"


def _proxy(mapping):
    return types.MappingProxyType(mapping)


def _reduce_proxy(proxy):
    return _proxy, (dict(proxy),)


@lru_cache(maxsize=1)
def _code_version():
    from importlib.metadata import version
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256(json.dumps([sys.version, *(version(name) for name in ("numpy", "scipy", "numba"))]).encode())
    for path in sorted(root.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            digest.update(str(path.relative_to(root)).encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()[:24]


def _cache_file(identity, sha256, reference_rate=DEFAULT_REFERENCE_RATE):
    key = [identity[0], sha256] if reference_rate == DEFAULT_REFERENCE_RATE else [identity[0], sha256, reference_rate]
    name = hashlib.sha256(json.dumps(key).encode()).hexdigest()
    return Path(os.environ[CONTEXT_CACHE_ENV]) / _code_version() / f"{name}.pkl"


def _save(path, context):
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with temporary.open("wb") as stream:
            pickler = pickle.Pickler(stream, protocol=5)
            pickler.dispatch_table = {**copyreg.dispatch_table, types.MappingProxyType: _reduce_proxy}
            pickler.dump(context)
        os.replace(temporary, path)
    except Exception as exc:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        print(f"context cache: {path.name} not stored ({exc!r})", file=sys.stderr)


def _load_or_build(identity, reference_rate=DEFAULT_REFERENCE_RATE):
    if not os.environ.get(CONTEXT_CACHE_ENV):
        return InstanceContext(identity, reference_rate)
    sha256 = hashlib.sha256(Path(identity[0]).read_bytes()).hexdigest()
    path = _cache_file(identity, sha256, reference_rate)
    if path.exists():
        collecting = gc.isenabled()
        gc.disable()
        try:
            with path.open("rb") as stream:
                context = pickle.load(stream)
            if context.path == identity[0] and context.sha256 == sha256:
                if context.reference_rate == context.tables.reference_rate == reference_rate:
                    return context
                print(f"context cache: {path.name} holds a {context.tables.reference_rate} context, "
                      f"{reference_rate} requested; rebuilding", file=sys.stderr)
        except Exception as exc:
            print(f"context cache: {path.name} unreadable ({exc!r}); rebuilding", file=sys.stderr)
        finally:
            if collecting:
                gc.enable()
    context = InstanceContext(identity, reference_rate)
    context.dp, context.reference
    _save(path, context)
    return context


_FREEZE = False


def _load(identity, reference_rate=DEFAULT_REFERENCE_RATE):
    context = _load_or_build(identity, reference_rate)
    if _FREEZE:
        gc.freeze()
    return context


def set_context_freeze(enabled):
    global _FREEZE
    _FREEZE = bool(enabled)


_context = lru_cache(maxsize=16)(_load)


def set_context_capacity(size):
    global _context
    if _context.cache_info().maxsize != size:
        _context = lru_cache(maxsize=size)(_load)


def get_context(instance_key, registry, reference_rate=None):
    entries = _registry(_identity(registry))
    if instance_key not in entries:
        raise KeyError(f"Unknown instance key: {instance_key}")
    rate = resolve_reference_rate(reference_rate)
    if rate == DEFAULT_REFERENCE_RATE:
        return _context(_identity(entries[instance_key]))
    return _context(_identity(entries[instance_key]), rate)


@lru_cache(maxsize=32)
def _grammar(context, zeta):
    from lawn_mec.vcp.grammar import Grammar
    from lawn_mec.vcp.payload import grammar_tables
    return Grammar(grammar_tables(context._public_payload), zeta=zeta)


@lru_cache(maxsize=8192)
def _check_prefix(grammar, prefix):
    if len(prefix) > len(grammar.fields):
        raise ValueError("Prefix extends beyond the V1 grammar")
    if prefix:
        _check_prefix(grammar, prefix[:-1])
        options = grammar.legal_options(prefix[:-1])
        if prefix[-1] not in options:
            raise ValueError(f"Illegal token {prefix[-1]!r} at field {len(prefix)-1}; legal={options}")


class VCPFieldGrammar:
    options = tuple(dict.fromkeys("ABCDEFGHLPQRSGXYZ-"))
    version = GRAMMAR_VERSION

    def __init__(self, *, registry, zeta=0.85):
        if not 0 < zeta <= 1:
            raise ValueError("zeta must be in (0, 1]")
        self.registry, self.zeta = str(registry), float(zeta)
        self.instance_key = None

    def reset(self, instance_key):
        context = get_context(instance_key, self.registry)
        if not hasattr(context, '_public_payload'):
            from lawn_mec.vcp.payload import prompt_payload
            context._public_payload = {k: v for k, v in prompt_payload(instance_key, self.registry).items()
                                       if k != 'context'}
        self.instance_key, self.context = instance_key, context
        self.tables, self.instance = context.tables, context.instance
        self.grammar = _grammar(context, self.zeta)
        self.blocks = len(context._public_payload['owners'])
        self.block_ends = tuple(
            max(i + 1 for i, (owner, _, _) in enumerate(self.grammar.fields) if owner == u)
            for u in range(self.blocks))

    def _prefix(self, prefix):
        if self.instance_key is None:
            raise RuntimeError("Call reset(instance_key) first")
        prefix = tuple(prefix)
        _check_prefix(self.grammar, prefix)
        return prefix

    def next_field(self, prefix):
        prefix = self._prefix(prefix)
        if len(prefix) == len(self.grammar.fields):
            return None
        u, k, field = self.grammar.fields[len(prefix)]
        options = self.grammar.legal_options(prefix)
        if not options:
            raise ValueError("V1 returned no continuation for a legal prefix")
        name = f"u{u}.layer" if k is None else f"u{u}.k{k}.{field}"
        return FieldSpec(name=name, options=options, forced=len(options) == 1)

    def render_prefix_text(self, prefix):
        prefix = self._prefix(prefix)
        if len(prefix) == len(self.grammar.fields):
            return ""
        u, k, field = self.grammar.fields[len(prefix)]
        if field == "layer":
            return f"\nU{u+1} layer "
        if field == "g":
            return f"\nT{k+1} "
        return " "

    def block_end(self, prefix, block):
        prefix = self._prefix(prefix)
        if not isinstance(block, int) or not 0 <= block < self.blocks:
            raise ValueError("block must be a zero-based UAV index")
        end = self.block_ends[block]
        if len(prefix) > end:
            raise ValueError("Prefix is past the requested UAV block")
        return end

    def decode(self, prefix):
        prefix = self._prefix(prefix)
        return self.grammar.decode(prefix)
