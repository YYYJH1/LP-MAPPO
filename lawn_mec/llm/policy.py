from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import torch

from .grammar import SequenceTrace
from .precision import configure, autocast

LORA_RANK = 64
LORA_ALPHA = 128
LORA_TARGETS = ('q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj')
LORA_DROPOUT = 0.0


def lora_record(rank=LORA_RANK, alpha=LORA_ALPHA, targets=LORA_TARGETS, dropout=LORA_DROPOUT):
    return dict(rank=int(rank), alpha=alpha, targets=list(targets), dropout=float(dropout))


@dataclass
class PolicyScore:
    field_logprobs: torch.Tensor
    field_kl: torch.Tensor
    field_entropy: torch.Tensor
    field_reference_logprobs: torch.Tensor | None = None

    @property
    def logprob(self):
        return self.field_logprobs.sum()


def restricted_logprobs(logits, allowed_ids):
    return torch.log_softmax(logits[..., list(allowed_ids)].float(), dim=-1)


def local_model_complete(path):
    import json
    from safetensors import SafetensorError, safe_open
    path = Path(path)
    if not (path / "config.json").is_file():
        return False
    index = path / "model.safetensors.index.json"
    try:
        json.loads((path / "config.json").read_text())
        weight_map = json.loads(index.read_text())["weight_map"] if index.exists() else None
        files = set(weight_map.values()) if weight_map else {"model.safetensors"}
        for filename in files:
            with safe_open(path / filename, framework="pt", device="cpu") as handle:
                keys = set(handle.keys())
                if weight_map and any(k not in keys for k, v in weight_map.items() if v == filename):
                    return False
        return True
    except (OSError, ValueError, KeyError, SafetensorError):
        return False


class LoRAPolicy:
    def __init__(self, model, tokenizer, *, reference_adapter="reference", lora=None):
        self.model, self.tokenizer = model, tokenizer
        self.reference_adapter = reference_adapter
        self.lora = lora

    @classmethod
    def from_base(cls, base, tokenizer, *, rank=LORA_RANK, alpha=LORA_ALPHA, targets=LORA_TARGETS,
                  gradient_checkpointing=True, precision='float16'):
        from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
        config = LoraConfig(task_type="CAUSAL_LM", r=rank, lora_alpha=alpha,
                            lora_dropout=LORA_DROPOUT, target_modules=list(targets), bias="none")
        model = get_peft_model(base.float(), config)
        reference = {k: v.detach().clone() for k, v in get_peft_model_state_dict(model).items()}
        ref_config = deepcopy(config)
        ref_config.inference_mode = True
        model.add_adapter("reference", ref_config)
        set_peft_model_state_dict(model, reference, adapter_name="reference")
        model.set_adapter("default")
        for name, parameter in model.named_parameters():
            if ".reference." in name:
                parameter.requires_grad_(False)
        for module in model.modules():
            if isinstance(module, torch.nn.Dropout):
                module.p = 0.0
            if hasattr(module, "attention_dropout"):
                module.attention_dropout = 0.0
        model.config.use_cache = False
        if gradient_checkpointing:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            model.enable_input_require_grads()
        configure(model, precision)
        return cls(model, tokenizer, lora=lora_record(rank, alpha, targets, LORA_DROPOUT))

    @classmethod
    def load(cls, path, *, device="cpu", dtype=torch.float32, **kwargs):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if dtype != torch.float32:
            raise ValueError('master dtype must be float32; use precision= for autocast')
        if not local_model_complete(path):
            raise ValueError(f"Incomplete local model weights: {path}")
        tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
        base = AutoModelForCausalLM.from_pretrained(
            path, local_files_only=True, dtype=dtype, attn_implementation="eager")
        return cls.from_base(base.to(device), tokenizer, **kwargs)

    def parameters(self):
        return (p for p in self.model.parameters() if p.requires_grad)

    def score(self, traces: list[SequenceTrace], *, with_reference=True) -> list[PolicyScore]:
        if not traces:
            return []
        device = next(self.model.parameters()).device
        pad = self.tokenizer.pad_token_id
        if pad is None:
            pad = self.tokenizer.eos_token_id
        if pad is None:
            raise ValueError("A padding or EOS token is required")
        width = max(len(t.token_ids) for t in traces)
        ids = torch.full((len(traces), width), pad, dtype=torch.long, device=device)
        mask = torch.zeros_like(ids)
        for row, trace in enumerate(traces):
            length = len(trace.token_ids)
            ids[row, :length] = torch.tensor(trace.token_ids, device=device)
            mask[row, :length] = 1
        arguments = dict(input_ids=ids, attention_mask=mask,
                         position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False)

        def gather(logits):
            return [[restricted_logprobs(logits[row, f.position - 1], f.allowed_ids)
                     for f in trace.fields] for row, trace in enumerate(traces)]

        reference = None
        was_training = self.model.training
        if with_reference:
            try:
                self.model.set_adapter(self.reference_adapter, inference_mode=True)
                self.model.eval()
                with torch.no_grad(), autocast(self.model):
                    reference = gather(self.model(**arguments).logits)
            finally:
                self.model.set_adapter("default")
                self.model.train(was_training)
        with autocast(self.model):
            logits = self.model(**arguments).logits
        distributions = gather(logits)
        results = []
        for row, trace in enumerate(traces):
            logps, kls, entropies = [], [], []
            for j, f in enumerate(trace.fields):
                logp = distributions[row][j]
                zero = logp.sum() * 0
                logps.append(zero if f.forced else logp[f.selected])
                entropies.append(zero if f.forced else -(logp.exp() * logp).sum())
                kls.append(zero if reference is None or f.forced else
                           (logp.exp() * (logp - reference[row][j])).sum())
            empty = logits[row, 0, :0].float()
            ref_logps = None
            if reference is not None:
                refs = [reference[row][j].sum() * 0 if f.forced else reference[row][j][f.selected]
                        for j, f in enumerate(trace.fields)]
                ref_logps = torch.stack(refs) if refs else empty.detach()
            results.append(PolicyScore(*(torch.stack(v) if v else empty
                                         for v in (logps, kls, entropies)), ref_logps))
        return results

    def score_packed(self, packing, nodes, *, with_reference=True) -> list[PolicyScore]:
        if not nodes:
            return []
        causal = self.model.get_base_model()
        implementation = causal.config._attn_implementation
        if implementation not in ('eager', 'sdpa'):
            raise ValueError('tree packing requires eager or SDPA attention')
        if getattr(causal.model, 'has_sliding_layers', False) or getattr(causal.config, 'use_sliding_window', False):
            raise ValueError('tree packing requires full attention in every layer')
        parameter = next(self.model.parameters())
        arguments, paths = packing.arguments(nodes, device=parameter.device, dtype=parameter.dtype)
        if implementation == 'sdpa':
            arguments['is_causal'] = False
        head = causal.get_output_embeddings()
        requests = {}
        keys = []
        for node in nodes:
            row = []
            for f in node.sample.trace.fields:
                key = (paths[node.index][f.position - 1], f.allowed_ids)
                requests.setdefault(key, len(requests))
                row.append(key)
            keys.append(row)

        def distributions():
            hidden = causal.model(**arguments).last_hidden_state[0]
            grouped = {}
            for position, allowed in requests:
                grouped.setdefault(allowed, []).append(position)
            values = {}
            for allowed, positions in grouped.items():
                ids = list(allowed)
                logits = torch.nn.functional.linear(hidden[positions], head.weight[ids],
                    None if head.bias is None else head.bias[ids])
                logps = torch.log_softmax(logits.float(), dim=-1)
                values.update({(position, allowed): logps[i] for i, position in enumerate(positions)})
            return values, hidden[0, :0].float()

        reference = None
        was_training = self.model.training
        if with_reference:
            try:
                self.model.set_adapter(self.reference_adapter, inference_mode=True)
                self.model.eval()
                with torch.no_grad(), autocast(self.model):
                    reference, _ = distributions()
            finally:
                self.model.set_adapter('default')
                self.model.train(was_training)
        with autocast(self.model):
            current, empty = distributions()
        results = []
        for node, row in zip(nodes, keys):
            logps, kls, entropies, refs = [], [], [], []
            for f, key in zip(node.sample.trace.fields, row):
                lp = current[key]
                zero = lp.sum() * 0
                logps.append(zero if f.forced else lp[f.selected])
                entropies.append(zero if f.forced else -(lp.exp() * lp).sum())
                kls.append(zero if reference is None or f.forced else
                           (lp.exp() * (lp - reference[key])).sum())
                if reference is not None:
                    refs.append(reference[key].sum() * 0 if f.forced else reference[key][f.selected])
            results.append(PolicyScore(*(torch.stack(v) if v else empty for v in (logps, kls, entropies)),
                                       None if reference is None else torch.stack(refs)))
        return results

    def tree_scores(self, tree, *, max_blocks=32, with_reference=True):
        from .tree_pack import TreePacking
        packing = TreePacking(tree, self.tokenizer)
        for nodes in packing.chunks(max_blocks):
            yield nodes, self.score_packed(packing, nodes, with_reference=with_reference)

    def save_adapter(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=False)
        self.model.save_pretrained(path, selected_adapters=["default"], safe_serialization=True)
        return path
