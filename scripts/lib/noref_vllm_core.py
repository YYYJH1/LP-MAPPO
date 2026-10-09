import hashlib
import json
import math
from dataclasses import dataclass, field


def context_hash(ids):
    return hashlib.sha256(json.dumps(list(ids), separators=(',', ':')).encode()).hexdigest()


def first_argmax(values):
    if not values or not all(math.isfinite(v) for v in values):
        raise FloatingPointError('missing/nonfinite allowed field scores')
    return max(range(len(values)), key=values.__getitem__)


@dataclass
class State:
    key: str
    grammar: object
    ids: list
    prefix: tuple = ()
    fields: list = field(default_factory=list)
    error: str | None = None


class FieldArgmax:
    def __init__(self, sampler, max_model_len):
        if sampler.path != 'field':
            raise ValueError('field engine required')
        self.sampler = sampler
        self.tokenizer, self.mapping = sampler.tokenizer, sampler.mapping
        self.spec = sampler.grammar_spec
        self.max_model_len = max_model_len

    def score(self, prompts, allowed):
        from vllm import SamplingParams
        params = [SamplingParams(n=1, temperature=0., top_p=1., top_k=-1,
                  max_tokens=1, allowed_token_ids=list(ids), logprobs=len(ids),
                  detokenize=False, seed=0) for ids in allowed]
        outputs = self.sampler.engine.generate(
            [{'prompt_token_ids': list(ids)} for ids in prompts], params,
            lora_request=self.sampler.lora_request, use_tqdm=False)
        if len(outputs) != len(prompts):
            raise RuntimeError('incomplete field batch')
        result = []
        for request, ids in zip(outputs, allowed):
            if len(request.outputs) != 1 or len(request.outputs[0].token_ids) != 1:
                raise RuntimeError('expected one completion with one field token')
            output = request.outputs[0]
            if output.token_ids[0] not in ids or not output.logprobs:
                raise RuntimeError('vLLM returned an unallowed token/missing scores')
            scores = output.logprobs[0]
            if any(token not in scores for token in ids):
                raise RuntimeError('vLLM did not return every allowed score')
            values = [float(scores[token].logprob) for token in ids]
            first_argmax(values)
            result.append(values)
        return result

    def generate(self, requests):
        states = [State(key, self.spec.build(key), list(self.tokenizer.encode(
                  prompt, add_special_tokens=False))) for key, prompt in requests]
        if any(not s.ids for s in states):
            raise ValueError('Prompt must contain at least one token')
        while True:
            active, prompts, allowed = [], [], []
            progressing = False
            for s in states:
                if s.error is not None:
                    continue
                try:
                    f = s.grammar.next_field(s.prefix)
                    if f is None:
                        continue
                    progressing = True
                    template = self.tokenizer.encode(s.grammar.render_prefix_text(s.prefix),
                                                     add_special_tokens=False)
                    ids = s.ids + list(template)
                    if len(ids) + 1 > self.max_model_len:
                        raise RuntimeError('full context exceeds --max-model-len for '+s.key)
                    tokens = self.mapping.ids(f)
                    if f.forced:
                        self.accept(s, f, ids, tokens, [0.])
                    else:
                        active.append((s, f, ids, tokens))
                        prompts.append(ids); allowed.append(tokens)
                except (ValueError, IndexError, KeyError, FloatingPointError) as exc:
                    s.error = str(exc)
            if not progressing:
                break
            if active:
                for (s, f, ids, tokens), scores in zip(active, self.score(prompts, allowed)):
                    self.accept(s, f, ids, tokens, scores)
        return states

    @staticmethod
    def accept(s, f, ids, tokens, scores):
        index = first_argmax(scores)
        selected = f.options[index]
        s.fields.append(dict(index=len(s.prefix), field=f.name, options=list(f.options),
            selected=selected, forced=f.forced, logprobs=scores, context_sha256=context_hash(ids)))
        s.ids = ids + [tokens[index]]
        s.prefix += (selected,)


def create(tokenizer, registry, adapter, *, memory=.25, max_model_len=16384):
    from lawn_mec.llm.grammar import CanonicalOptions, DESIGN_OPTIONS
    from lawn_mec.llm.sampler import VLLMSampler
    from noref_input import spec
    mapping = CanonicalOptions.build(tokenizer, DESIGN_OPTIONS)
    sampler = VLLMSampler.create(str(__import__('noref_common').MODEL), mapping, spec(registry),
        path='field', gpu_memory_utilization=memory, max_model_len=max_model_len,
        max_lora_rank=64, dtype='float16')
    try:
        sampler.mapping.validate(tokenizer)
        sampler.load_adapter(adapter, 1)
        return sampler, FieldArgmax(sampler, max_model_len)
    except BaseException:
        sampler.close()
        raise
