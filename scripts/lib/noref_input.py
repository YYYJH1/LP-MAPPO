import math
import numpy as np
from lawn_mec.vcp.features import static_features
from lawn_mec.vcp.payload import SCHEMA, number, boundaries, _runs, _intervals, grammar_tables
from lawn_mec.vcp.grammar import Grammar
from lawn_mec.llm.vcp_grammar import VCPFieldGrammar, get_context
from lawn_mec.llm.prompt_compact import serialize_prompt


def payload(ctx):
    tb = ctx.tables
    f = static_features(tb, ctx.dp, reference=None)
    bs = [("PQR"[m], (_intervals(np.flatnonzero(caps == np.min(caps))),
                     number(np.min(caps)/tb.p['delta']/1e9), number(np.max(caps)/tb.p['delta']/1e9)))
          for m, caps in enumerate(tb.cpu_capacity)]
    uavs, tasks = [], []
    for u, own in enumerate(tb.own):
        los = np.mean(f['line_of_sight'][list(own)], axis=0).T
        uavs.append((f'U{u+1}', (number(tb.p['battery']/1000), number(f['free_energy_J'][u]/1000),
                    number(f['all_keep_energy_J'][u]/1000), _runs(['/'.join(number(x) for x in row) for row in los], 'ABCDEFG'))))
        for k in own:
            t = tb.tasks[k]; layers = np.argmax(f['line_of_sight'][k], axis=1)
            upload = [f['upload_duration'][k, m, ell] for m, ell in enumerate(layers)]
            sight = [f['line_of_sight'][k, m, ell] for m, ell in enumerate(layers)]
            tasks.append((f'T{k+1}', (boundaries(f['visit_bins'][k, 0]), boundaries(f['visit_bins'][k, 1]),
                str(math.floor(t['d']/tb.p['delta'])), str(math.floor(t['H']/tb.p['delta'])),
                number(t['D']/1e6), number(t['C']/1e9), str(int(f['local_duration'][k])),
                '/'.join(str(int(x)) for x in upload), ''.join('L' if x else 'N' for x in sight),
                '/'.join(str(int(x)) for x in f['mec_duration'][k]), number(f['rush_marginal_J'][k]/1000),
                boundaries(f['upload_bins'][k]), ''.join('ABCDEFG'[ell] for ell in layers))))
    legal = dict(M=int(tb.p['M']), U=int(tb.p['U']), N=int(tb.N), own=[list(o) for o in tb.own],
        visits=tb.instance['visits'], tasks=[dict(u=int(t['u']), L_s=int(t['L_s']), L_a=int(t['L_a']),
          z_s=int(t['z_s']), z_a=int(t['z_a']), d=math.floor(t['d']/tb.p['delta']), H=math.floor(t['H']/tb.p['delta'])) for t in tb.tasks],
        visit_bins=[[[list(pair) for pair in tb.visit_bins[k, phase]] for phase in ('s','a')] for k in range(len(tb.tasks))],
        upload_bins=[[list(pair) for pair in tb.upload_bins[k]] for k in range(len(tb.tasks))],
        ready_min=tb.ready_min.tolist(), min_time=tb.min_time.tolist())
    return dict(tables=(bs,uavs,tasks), schema=SCHEMA, instance_sha256=ctx.sha256,
        c_min=Grammar(tb).c_min, K=len(tb.tasks), g_cap=len(tb.tasks)-Grammar(tb).c_min,
        owners=tuple(tuple(f'T{k+1}' for k in own) for own in tb.own), legal=legal)


def build_prompt(tokenizer, key, registry):
    return serialize_prompt(tokenizer, payload(get_context(key, registry)), variant='no_reference')


class NoReferenceGrammar(VCPFieldGrammar):
    def reset(self, instance_key):
        ctx = get_context(instance_key, self.registry)
        if not hasattr(ctx, '_noref_grammar'):
            ctx._noref_grammar = Grammar(grammar_tables(payload(ctx)), zeta=self.zeta)
        self.instance_key, self.context = instance_key, ctx
        self.tables, self.instance, self.grammar = ctx.tables, ctx.instance, ctx._noref_grammar
        self.blocks = len(ctx.tables.own)
        self.block_ends = tuple(max(i+1 for i,(owner,_,_) in enumerate(self.grammar.fields) if owner == u) for u in range(self.blocks))


def spec(registry):
    from lawn_mec.llm.grammar import GrammarSpec
    return GrammarSpec('noref_input:NoReferenceGrammar', dict(registry=str(registry), zeta=.85))


def traces_for(policy, records, registry, max_length=8192):
    from lawn_mec.llm.grammar import CanonicalOptions, DESIGN_OPTIONS, encode_trace
    mapping = CanonicalOptions.build(policy.tokenizer, DESIGN_OPTIONS); grammar = spec(registry)
    result = []
    for r in records:
        prompt = build_prompt(policy.tokenizer, r['instance_key'], registry)
        if prompt != r['prompt']:
            raise ValueError('prepared prompt/tokenizer/source changed')
        trace = encode_trace(policy.tokenizer, mapping, grammar, r['instance_key'], prompt, tuple(r['tokens']))
        if grammar.build(r['instance_key']).next_field(trace.prefix) is not None or len(trace.token_ids) > max_length:
            raise ValueError('incomplete or overlength trace; no truncation')
        result.append(trace)
    return result
