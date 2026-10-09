from dataclasses import asdict, dataclass
import gc
import hashlib
import json
import os

from .grammar import CanonicalOptions, FieldTrace, GrammarSpec, SequenceTrace, TokenMachine, encode_trace


def derive_seed(seed, *address):
    payload = json.dumps([seed, *address], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "little") % (2**63 - 1)


@dataclass(frozen=True)
class Sample:
    trace: SequenceTrace
    start_field: int
    field_logprobs: tuple[float, ...]
    generated_tokens: int
    mode: bool = False

    @property
    def logprob(self):
        return sum(self.field_logprobs)


REQUEST_FIELDS = ("instance_key", "prompt", "prefix", "stop_after_fields", "n", "mode", "seed")
MAX_FIELD_PROMPTS = 2048


def sample_requests(sampler, requests):
    requests = [tuple(request) for request in requests]
    if any(len(request) != len(REQUEST_FIELDS) for request in requests):
        raise ValueError(f"a sample request holds {REQUEST_FIELDS}")
    many = getattr(sampler, "sample_many", None)
    if many is not None:
        results = many(requests)
    else:
        results = [sampler.sample(key, prompt, prefix=prefix, stop_after_fields=stop, n=n, mode=mode, seed=seed)
                   for key, prompt, prefix, stop, n, mode, seed in requests]
    if len(results) != len(requests):
        raise RuntimeError("the sampler answered a different number of requests")
    return results


def engine_kv_tokens(engine):
    try:
        cache = engine.llm_engine.vllm_config.cache_config
        blocks, size = cache.num_gpu_blocks, cache.block_size
    except AttributeError:
        return None
    ok = all(isinstance(v, int) and not isinstance(v, bool) and v > 0 for v in (blocks, size))
    return blocks * size if ok else None


class _FieldRequest:
    def __init__(self, key, prompt, prefix, stop, n, mode, seed, initial):
        self.key, self.prompt, self.prefix, self.stop = key, prompt, prefix, stop
        self.n, self.mode, self.seed = n, mode, seed
        self.prefixes = [prefix for _ in range(n)]
        self.logprobs = [[] for _ in range(n)]
        self.emitted = [0] * n
        self.ids = [list(initial.token_ids) for _ in range(n)]
        self.fields = [list(initial.fields) for _ in range(n)]
        self.round_index = 0
        self.done = False

    def append(self, i, spec, template, selected, logprob, mapping):
        option = spec.options[selected]
        ids = self.ids[i]
        ids.extend(template)
        self.fields[i].append(FieldTrace(spec.name, len(ids), mapping.ids(spec), selected, spec.forced))
        ids.append(mapping.token_ids[option])
        self.prefixes[i] += (option,)
        self.logprobs[i].append(logprob)

    def samples(self):
        return [Sample(SequenceTrace(self.key, self.prompt, p, tuple(ids), tuple(fields)),
                       len(self.prefix), tuple(lp), count, self.mode)
                for p, ids, fields, lp, count in zip(self.prefixes, self.ids, self.fields,
                                                     self.logprobs, self.emitted)]


def _frozen_reduced_precision(model):
    params = [p for p in model.parameters() if p.is_floating_point()]
    return (not hasattr(model, '_planner_precision') and not any(p.requires_grad for p in params)
            and any(str(p.dtype) != 'torch.float32' for p in params))


class TransformersSampler:
    exact_resume = True

    def __init__(self, model, tokenizer, mapping, grammar_spec):
        self.model, self.tokenizer = model, tokenizer
        self.mapping, self.grammar_spec = mapping, grammar_spec
        from .precision import contract, normalize
        self.frozen_storage = _frozen_reduced_precision(model)
        if self.frozen_storage:
            self.precision = normalize(next(p.dtype for p in model.parameters() if p.is_floating_point()))
        else:
            self.precision = contract(model)['compute']

    def sample(self, instance_key, prompt, *, prefix=(), stop_after_fields=None,
               n=1, mode=False, seed=None):
        import torch
        from .policy import restricted_logprobs
        if n < 1:
            raise ValueError("n must be positive")
        TokenMachine(self.tokenizer, self.mapping, self.grammar_spec,
                     instance_key, prefix, stop_after_fields)
        initial = encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                               instance_key, prompt, tuple(prefix))
        prefixes, logs = [tuple(prefix) for _ in range(n)], [[] for _ in range(n)]
        generator = torch.Generator(device="cpu")
        if seed is not None:
            generator.manual_seed(seed)
        else:
            generator.seed()
        device = next(self.model.parameters()).device
        training = self.model.training
        self.model.eval()
        try:
            from contextlib import nullcontext
            from .precision import autocast
            with torch.no_grad(), (nullcontext() if self.frozen_storage else autocast(self.model)):
                while True:
                    active, contexts = [], []
                    for i, p in enumerate(prefixes):
                        grammar = self.grammar_spec.build(instance_key)
                        f = grammar.next_field(p)
                        if f is None or len(p) == stop_after_fields:
                            continue
                        if f.forced:
                            prefixes[i] += (f.options[0],)
                            logs[i].append(0.0)
                            continue
                        trace = encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                                             instance_key, prompt, p)
                        contexts.append(list(trace.token_ids) + self.tokenizer.encode(
                            grammar.render_prefix_text(p), add_special_tokens=False))
                        active.append((i, f))
                    if not active:
                        if all((len(p) == stop_after_fields or
                                self.grammar_spec.build(instance_key).next_field(p) is None)
                               for p in prefixes):
                            break
                        continue
                    width = max(map(len, contexts))
                    pad = self.tokenizer.pad_token_id
                    if pad is None:
                        pad = self.tokenizer.eos_token_id
                    ids = torch.full((len(active), width), pad, device=device, dtype=torch.long)
                    mask = torch.zeros_like(ids)
                    for row, context in enumerate(contexts):
                        ids[row, :len(context)] = torch.tensor(context, device=device)
                        mask[row, :len(context)] = 1
                    logits = self.model(input_ids=ids, attention_mask=mask,
                        position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False).logits
                    for row, (i, f) in enumerate(active):
                        lp = restricted_logprobs(logits[row, len(contexts[row]) - 1], self.mapping.ids(f))
                        selected = int(lp.argmax()) if mode else int(torch.multinomial(
                            lp.exp().cpu(), 1, generator=generator))
                        prefixes[i] += (f.options[selected],)
                        logs[i].append(float(lp[selected]))
        finally:
            self.model.train(training)
        traces = [encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                               instance_key, prompt, p) for p in prefixes]
        return [Sample(t, len(prefix), tuple(lp), len(t.token_ids) - len(initial.token_ids), mode)
                for t, lp in zip(traces, logs)]

    def sample_many(self, requests):
        return [self.sample(key, prompt, prefix=prefix, stop_after_fields=stop, n=n, mode=mode, seed=seed)
                for key, prompt, prefix, stop, n, mode, seed in requests]

    def load_adapter(self, adapter_path, version):
        from peft import set_peft_model_state_dict
        from peft.utils.save_and_load import load_peft_weights
        state = load_peft_weights(str(adapter_path), device="cpu", local_files_only=True)
        set_peft_model_state_dict(self.model, state, adapter_name="default")


class VLLMSampler:
    exact_resume = False
    max_field_prompts = MAX_FIELD_PROMPTS
    wave_tokens = 'auto'
    wave_fraction = 0.75

    def __init__(self, engine, tokenizer, mapping, grammar_spec, *, path="fsm",
                 max_tokens=1024, owns_engine=False, precision='float16'):
        if path not in ("fsm", "field"):
            raise ValueError("path must be fsm or field")
        self.engine, self.tokenizer = engine, tokenizer
        self.mapping, self.grammar_spec = mapping, grammar_spec
        self.path, self.max_tokens, self.owns_engine = path, max_tokens, owns_engine
        self.lora_request = None
        from .precision import normalize
        self.precision = normalize(precision)

    @classmethod
    def create(cls, model_path, mapping, grammar_spec, *, path="fsm",
               gpu_memory_utilization=0.35, max_model_len=2048, max_lora_rank=64,
               dtype="float16"):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if len(visible) != 1 or not visible[0] or visible[0] == "-1":
            raise RuntimeError("Select exactly one GPU using CUDA_VISIBLE_DEVICES first")
        from transformers import AutoTokenizer
        from vllm import LLM
        tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        mapping.validate(tokenizer)
        engine = None
        try:
            engine = LLM(
                model=str(model_path), tokenizer=str(model_path), trust_remote_code=False,
                generation_config="vllm",
                tensor_parallel_size=1, dtype=dtype, enforce_eager=True,
                gpu_memory_utilization=gpu_memory_utilization, max_model_len=max_model_len,
                enable_prefix_caching=True, enable_lora=True, max_loras=1,
                max_lora_rank=max_lora_rank, logprobs_mode="processed_logprobs",
                logits_processors=["lawn_mec.llm.logits_processor:FieldLogitsProcessor"]
                if path == "fsm" else None,
            )
            return cls(engine, tokenizer, mapping, grammar_spec, path=path, owns_engine=True, precision=dtype)
        except BaseException:
            if engine is not None:
                engine.llm_engine.engine_core.shutdown()
            raise

    def sample(self, instance_key, prompt, *, prefix=(), stop_after_fields=None,
               n=1, mode=False, seed=None):
        if n < 1:
            raise ValueError("n must be positive")
        prefix = tuple(prefix)
        TokenMachine(self.tokenizer, self.mapping, self.grammar_spec,
                     instance_key, prefix, stop_after_fields)
        if self.path == "field":
            return self._field(instance_key, prompt, prefix, stop_after_fields, n, mode, seed)
        from vllm import SamplingParams
        payload = dict(grammar=asdict(self.grammar_spec), mapping=asdict(self.mapping),
                       instance_key=instance_key, prefix=list(prefix),
                       stop_after_fields=stop_after_fields)
        params = SamplingParams(n=n, temperature=0.0 if mode else 1.0, top_p=1.0,
                                max_tokens=self.max_tokens, logprobs=0,
                                detokenize=False, seed=seed,
                                extra_args={"llm_field_grammar": json.dumps(payload)})
        initial = encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                               instance_key, prompt, prefix)
        outputs = self.engine.generate([{"prompt_token_ids": list(initial.token_ids)}],
                                       params, lora_request=self.lora_request, use_tqdm=False)
        samples = []
        for output in outputs[0].outputs:
            machine = TokenMachine(self.tokenizer, self.mapping, self.grammar_spec,
                                   instance_key, prefix, stop_after_fields)
            field_logprobs = []
            for i, token in enumerate(output.token_ids):
                if machine.at_field:
                    value = float(output.logprobs[i][token].logprob)
                    field_logprobs.append(0.0 if machine.field.forced else value)
                machine.consume(token)
            if machine.field is not None:
                raise RuntimeError("Generation truncated before grammar/block completed")
            trace = encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                                 instance_key, prompt, machine.prefix)
            samples.append(Sample(trace, len(prefix), tuple(field_logprobs),
                                  len(output.token_ids), mode))
        if len(samples) != n:
            raise RuntimeError("vLLM returned an unexpected number of samples")
        return samples

    def sample_many(self, requests):
        requests = [tuple(request) for request in requests]
        if any(len(request) != len(REQUEST_FIELDS) for request in requests):
            raise ValueError(f"a sample request holds {REQUEST_FIELDS}")
        if self.path != "field":
            return [self.sample(key, prompt, prefix=prefix, stop_after_fields=stop, n=n, mode=mode, seed=seed)
                    for key, prompt, prefix, stop, n, mode, seed in requests]
        states = []
        for key, prompt, prefix, stop, n, mode, seed in requests:
            if n < 1:
                raise ValueError("n must be positive")
            prefix = tuple(prefix)
            TokenMachine(self.tokenizer, self.mapping, self.grammar_spec, key, prefix, stop)
            initial = encode_trace(self.tokenizer, self.mapping, self.grammar_spec, key, prompt, prefix)
            states.append(_FieldRequest(key, prompt, prefix, stop, n, mode, seed, initial))
        for wave in self._waves(states):
            self._field_wave(wave)
        return [state.samples() for state in states]

    def _wave_budget(self):
        if self.wave_tokens is None:
            return None
        if self.wave_tokens == 'auto':
            capacity = engine_kv_tokens(self.engine)
            return None if capacity is None else max(1, int(self.wave_fraction * capacity))
        if isinstance(self.wave_tokens, bool) or not isinstance(self.wave_tokens, int) or self.wave_tokens < 1:
            raise ValueError("wave_tokens must be 'auto', None or a positive integer")
        return self.wave_tokens

    def _waves(self, states):
        budget = self._wave_budget()
        if budget is None or len(states) < 2:
            return [states]
        try:
            block = int(self.engine.llm_engine.vllm_config.cache_config.block_size)
        except (AttributeError, TypeError, ValueError):
            block = 16
        prompt_tokens = {}
        waves, wave, used, shared = [], [], 0, set()
        for state in states:
            if state.prompt not in prompt_tokens:
                prompt_tokens[state.prompt] = len(self.tokenizer.encode(state.prompt, add_special_tokens=False))
            table = self._field_tokens(state.key)
            end = len(table) - 1 if state.stop is None else min(state.stop, len(table) - 1)
            growth = table[end] - table[min(len(state.prefix), end)]
            prompt = prompt_tokens[state.prompt]
            own = len(state.ids[0]) - prompt + state.n * (growth + block)
            need = own + (0 if (state.key, state.prompt) in shared else prompt)
            if wave and used + need > budget:
                waves.append(wave)
                wave, used, shared = [], 0, set()
                need = own + prompt
            wave.append(state)
            used += need
            shared.add((state.key, state.prompt))
        waves.append(wave)
        return waves

    def _field_tokens(self, key):
        tables = self.__dict__.setdefault("_field_token_tables", {})
        table = tables.get(key)
        if table is None:
            grammar, table, prefix = self.grammar_spec.build(key), [0], ()
            while table[-1] <= self.max_tokens:
                spec = grammar.next_field(prefix)
                if spec is None:
                    break
                table.append(table[-1] + len(self.tokenizer.encode(grammar.render_prefix_text(prefix),
                                                                   add_special_tokens=False)) + 1)
                prefix += (spec.options[0],)
            tables[key] = table
        return table

    def _field_wave(self, wave):
        from vllm import SamplingParams
        grammars = {}
        for state in wave:
            if state.key not in grammars:
                grammars[state.key] = self.grammar_spec.build(state.key)
        while True:
            prompts, params, active, advanced = [], [], [], []
            for state in wave:
                if state.done:
                    continue
                grammar = grammars[state.key]
                any_field = False
                for i in range(state.n):
                    spec = grammar.next_field(state.prefixes[i])
                    if spec is None or len(state.prefixes[i]) == state.stop:
                        continue
                    any_field = True
                    template = self.tokenizer.encode(grammar.render_prefix_text(state.prefixes[i]),
                                                     add_special_tokens=False)
                    state.emitted[i] += len(template) + 1
                    if state.emitted[i] > self.max_tokens:
                        raise RuntimeError("Generation exceeded max_tokens")
                    if spec.forced:
                        state.append(i, spec, template, 0, 0.0, self.mapping)
                        continue
                    prompts.append({"prompt_token_ids": state.ids[i] + template})
                    params.append(SamplingParams(
                        temperature=0.0 if state.mode else 1.0, top_p=1.0, max_tokens=1,
                        allowed_token_ids=list(self.mapping.ids(spec)), logprobs=0,
                        detokenize=False,
                        seed=None if state.seed is None else derive_seed(state.seed, "field", state.round_index, i)))
                    active.append((state, i, spec, template))
                if any_field:
                    advanced.append(state)
                else:
                    state.done = True
            if not advanced:
                return
            for start in range(0, len(prompts), self.max_field_prompts):
                end = start + self.max_field_prompts
                outputs = self.engine.generate(prompts[start:end], params[start:end],
                                               lora_request=self.lora_request, use_tqdm=False)
                if len(outputs) != len(active[start:end]):
                    raise RuntimeError("Incomplete per-field batch")
                for (state, i, spec, template), result in zip(active[start:end], outputs):
                    output = result.outputs[0]
                    if len(output.token_ids) != 1:
                        raise RuntimeError("Expected exactly one field token")
                    token = output.token_ids[0]
                    state.append(i, spec, template, self.mapping.ids(spec).index(token),
                                 float(output.logprobs[0][token].logprob), self.mapping)
            for state in advanced:
                state.round_index += 1

    def _field(self, key, prompt, prefix, stop, n, mode, seed):
        from vllm import SamplingParams
        prefixes = [prefix for _ in range(n)]
        logprobs = [[] for _ in range(n)]
        emitted = [0] * n
        round_index = 0
        while True:
            prompts, params, active = [], [], []
            any_field = False
            for i in range(n):
                grammar = self.grammar_spec.build(key)
                spec = grammar.next_field(prefixes[i])
                if spec is None or len(prefixes[i]) == stop:
                    continue
                any_field = True
                template = self.tokenizer.encode(grammar.render_prefix_text(prefixes[i]),
                                                 add_special_tokens=False)
                emitted[i] += len(template) + 1
                if emitted[i] > self.max_tokens:
                    raise RuntimeError("Generation exceeded max_tokens")
                if spec.forced:
                    prefixes[i] += (spec.options[0],)
                    logprobs[i].append(0.0)
                    continue
                trace = encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                                     key, prompt, prefixes[i])
                prompts.append({"prompt_token_ids": list(trace.token_ids) + template})
                params.append(SamplingParams(
                    temperature=0.0 if mode else 1.0, top_p=1.0, max_tokens=1,
                    allowed_token_ids=list(self.mapping.ids(spec)), logprobs=0,
                    detokenize=False, seed=None if seed is None else derive_seed(seed, "field", round_index, i)))
                active.append((i, spec))
            if not any_field:
                break
            if active:
                outputs = self.engine.generate(prompts, params,
                                               lora_request=self.lora_request, use_tqdm=False)
                if len(outputs) != len(active):
                    raise RuntimeError("Incomplete per-field batch")
                for (i, spec), result in zip(active, outputs):
                    output = result.outputs[0]
                    if len(output.token_ids) != 1:
                        raise RuntimeError("Expected exactly one field token")
                    token = output.token_ids[0]
                    option = spec.options[self.mapping.ids(spec).index(token)]
                    prefixes[i] += (option,)
                    logprobs[i].append(float(output.logprobs[0][token].logprob))
            round_index += 1
        return [Sample(encode_trace(self.tokenizer, self.mapping, self.grammar_spec,
                                    key, prompt, p), len(prefix), tuple(lp), count, mode)
                for p, lp, count in zip(prefixes, logprobs, emitted)]

    def load_adapter(self, adapter_path, version):
        from vllm.lora.request import LoRARequest
        if version <= 0 or (self.lora_request and version <= self.lora_request.lora_int_id):
            raise ValueError("LoRA versions must be positive and strictly increasing")
        if self.lora_request is not None:
            if not self.engine.llm_engine.remove_lora(self.lora_request.lora_int_id):
                raise RuntimeError("Could not remove previous LoRA")
            self.lora_request = None
        self.engine.reset_prefix_cache()
        request = LoRARequest(f"policy-{version}", version, str(adapter_path))
        if not self.engine.llm_engine.add_lora(request):
            raise RuntimeError("vLLM rejected LoRA load")
        self.lora_request = request

    def close(self):
        if self.engine is None:
            return
        engine, self.engine = self.engine, None
        try:
            if self.owns_engine:
                engine.llm_engine.engine_core.shutdown()
        finally:
            del engine
            gc.collect()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
