from contextlib import contextmanager
from pathlib import Path
import fcntl
import hashlib
import json
import os
import sys

LIB = Path(__file__).resolve().parent
CODE = LIB.parents[1]
sys.path.insert(0, str(CODE))
THREADS = ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')
for _var in THREADS:
    os.environ.setdefault(_var, '8')
DATA = Path(os.environ.get('LPM_DATA_ROOT', CODE/'data')).resolve()
TASK = Path(os.environ.get('LPM_RUNS_ROOT', CODE/'runs')).resolve()
ROOT = TASK
SPLITS = DATA/'instances/splits'
RUNS = DATA/'checkpoints'
MODEL = Path(os.environ.get('LPM_MODEL', CODE/'models/qwen3-1.7b')).resolve()
SEEDS = (42, 43, 44)
MAPPOS = (42, 43, 44, 45, 46)
NOISE = {'tune': (910000, 911000), 'validation': (912000, 913000),
         'test': (1010000, 1011000)}
SCREEN = {'test': 1060000}
EXPECTED = {'train': 1747, 'tune': 128, 'validation': 128, 'test': 256}
VERSION = '2dcac95d14680b1afa3d513d'


def add_path_args(parser, *, model=False):
    parser.add_argument('--data-root', default=str(DATA), help='input data directory (default: <repo>/data)')
    parser.add_argument('--runs-root', default=str(TASK),
                        help='root for outputs and runtime state such as built contexts (default: <repo>/runs)')
    if model:
        parser.add_argument('--model', default=str(MODEL),
                            help='local Qwen3-1.7B directory (default: <repo>/models/qwen3-1.7b; not included)')
    return parser


def configure(args):
    global DATA, TASK, ROOT, SPLITS, RUNS, MODEL
    DATA = Path(args.data_root).resolve(); TASK = ROOT = Path(args.runs_root).resolve()
    SPLITS = DATA/'instances/splits'; RUNS = DATA/'checkpoints'
    os.environ.update(LPM_DATA_ROOT=str(DATA), LPM_RUNS_ROOT=str(TASK))
    if getattr(args, 'model', None):
        MODEL = Path(args.model).resolve(); os.environ['LPM_MODEL'] = str(MODEL)
    return args


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def output(path):
    p = Path(path).resolve()
    if p.is_relative_to(DATA):
        raise ValueError('outputs may not be written inside the data directory: ' + str(p))
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path, value):
    p = output(path); tmp = p.with_name(p.name+'.part')
    with tmp.open('w') as f:
        f.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+'\n')
        f.flush(); os.fsync(f.fileno())
    tmp.replace(p)


def pin(path, value):
    value = json.loads(json.dumps(value, allow_nan=False)); p = output(path)
    if p.exists() and json.loads(p.read_text()) != value:
        raise ValueError('resume input/config changed; use fresh output: ' + str(p))
    if not p.exists():
        write_json(p, value)
    return digest(value)


def rows(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]


def append(path, value):
    with output(path).open('a') as f:
        f.write(json.dumps(value, sort_keys=True, allow_nan=False)+'\n'); f.flush(); os.fsync(f.fileno())


@contextmanager
def lease(path):
    with output(path).open('a') as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def runtime(gpu=None):
    if os.getpriority(os.PRIO_PROCESS, 0) < 19:
        os.nice(19-os.getpriority(os.PRIO_PROCESS, 0))
    for var in THREADS:
        if os.environ.get(var) != '8':
            raise ValueError(var+'=8 required (these scripts use eight library threads per process)')
    for var, child in (('TMPDIR',''), ('TMP',''), ('TEMP',''), ('NUMBA_CACHE_DIR','numba'),
                       ('HF_HOME','hf'), ('XDG_CACHE_HOME','xdg'), ('TORCH_HOME','torch'),
                       ('TRITON_CACHE_DIR','triton'), ('TORCHINDUCTOR_CACHE_DIR','inductor'),
                       ('CUDA_CACHE_PATH','cuda')):
        d = TASK/'runtime'/child; d.mkdir(parents=True, exist_ok=True); os.environ[var] = str(d)
    os.environ.update(PYTHONDONTWRITEBYTECODE='1', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_HUB_DISABLE_TELEMETRY='1', DO_NOT_TRACK='1')
    if gpu is None:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
    else:
        if os.environ.get('CUDA_VISIBLE_DEVICES', str(gpu)) != str(gpu):
            raise ValueError('--gpu disagrees with CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)
    os.environ.setdefault('LAWN_CONTEXT_CACHE', str(TASK/'runtime/context_store'))
    from lawn_mec.llm import vcp_grammar as vg
    vg.set_reference_rate('nominal'); vg.set_context_capacity(1)


def manifest(split):
    if split not in EXPECTED:
        raise ValueError('unknown split')
    p = SPLITS/f'{split}.json'
    if not p.is_file():
        raise FileNotFoundError(f'{p} not found; create the {split} split first: '
                                f'python scripts/make_instances.py --split {split} --out {DATA}/instances')
    raw = json.loads(p.read_text())
    if raw.get('split') != split or len(raw['instances']) != EXPECTED[split]:
        raise ValueError('canonical split count/name differs')
    rr = []
    for orig in raw['instances']:
        r = dict(orig); q = Path(r['path']); r['path'] = str((q if q.is_absolute() else p.parent/q).resolve()); rr.append(r)
    if len({r['sha256'] for r in rr}) != len(rr):
        raise ValueError('duplicate instance hash')
    return dict(raw, instances=rr, manifest_sha256=sha(p), manifest_path=str(p))


def cache_paths(identity, h, rate='nominal'):
    from lawn_mec.llm import vcp_grammar as vg
    os.environ.setdefault('LAWN_CONTEXT_CACHE', str(TASK/'runtime/context_store'))
    return [vg._cache_file(identity, h, rate)]


def cached_path(r):
    from lawn_mec.llm import vcp_grammar as vg
    return next((p for p in cache_paths(vg._identity(r['path']), r['sha256']) if p.is_file()), None)


def read_cached(identity, reference_rate='nominal'):
    if reference_rate != 'nominal':
        raise ValueError('only the nominal reference rate is supported')
    from lawn_mec.llm import vcp_grammar as vg
    h = sha(identity[0])
    os.environ.setdefault('LAWN_CONTEXT_CACHE', str(TASK/'runtime/context_store'))
    ctx = vg._load_or_build(identity, 'nominal')
    if (ctx.path != identity[0] or ctx.sha256 != h or ctx.reference_rate != 'nominal'
            or ctx.tables.reference_rate != 'nominal'):
        raise ValueError('cache identity/rate differs')
    return ctx


def context(r):
    from lawn_mec.llm import vcp_grammar as vg
    if sha(r['path']) != r['sha256']:
        raise ValueError('instance hash differs')
    return vg.get_context(r['key'], registry_for_row(r))


def registry_for_row(r):
    p = TASK/'runtime'/f'registry-{os.getpid()}.json'
    value = {r['key']: r['path']}
    if not p.exists() or json.loads(p.read_text()) != value:
        write_json(p, value)
    return str(p)


def validate(ctx, plan):
    from lawn_mec.vcpm.library import pack, unpack
    from lawn_mec.vcp.grammar import Grammar
    if plan is None:
        return dict(valid=False, grammar_valid=False, precheck_passed=False, reason='not_decoded')
    try:
        obj = unpack(plan)
        if digest(pack(obj)) != digest(plan):
            raise ValueError('noncanonical packed structure/types')
        g = Grammar(ctx.tables); tokens = obj.tokens(ctx.tables)
        if not g.validate(tokens)['valid'] or digest(pack(g.decode(tokens))) != digest(plan):
            return dict(valid=False, grammar_valid=False, precheck_passed=False, reason='grammar')
        ok = bool(ctx.dp.precheck(obj.keep_sets(ctx.tables))['passed'])
        return dict(valid=ok, grammar_valid=True, precheck_passed=ok, reason='passed' if ok else 'precheck')
    except (ValueError, TypeError, KeyError, IndexError, AttributeError) as e:
        return dict(valid=False, grammar_valid=False, precheck_passed=False, reason='syntax: '+str(e))


def execution(plan, check, interface='R'):
    return (plan, interface) if check['valid'] else (None, 'L')


def agreement(ctx, plan):
    from lawn_mec.vcpm.library import pack, unpack
    from lawn_mec.vcp.grammar import Grammar
    ref = pack(ctx.reference); expected = ctx.reference.tokens(ctx.tables)
    try:
        actual = unpack(plan).tokens(ctx.tables) if plan is not None else ()
    except (ValueError, TypeError, KeyError, IndexError, AttributeError):
        actual = ()
    fields = Grammar(ctx.tables).fields
    matched = [i < len(actual) and a == actual[i] for i, a in enumerate(expected)]
    by = {}
    for f, ok in zip(fields, matched):
        name = f[2]; item = by.setdefault(name, dict(correct=0, total=0)); item['correct'] += int(ok); item['total'] += 1
    return dict(exact=plan is not None and digest(plan) == digest(ref), correct=sum(matched), total=len(expected), by_field=by)


def stats(records):
    n = len(records)
    by = {}
    for r in records:
        for f, v in r['agreement']['by_field'].items():
            x = by.setdefault(f, dict(correct=0, total=0)); x['correct'] += v['correct']; x['total'] += v['total']
    return dict(instances=n, legal=sum(r['check']['valid'] for r in records), invalid=sum(not r['check']['valid'] for r in records),
                legal_rate=sum(r['check']['valid'] for r in records)/n,
                exact_dp_rate=sum(r['agreement']['exact'] for r in records)/n,
                field_accuracy=sum(r['agreement']['correct'] for r in records)/sum(r['agreement']['total'] for r in records),
                by_field={f: dict(v, accuracy=v['correct']/v['total']) for f, v in by.items()}, invalid_execution='L/f17; no DP')
