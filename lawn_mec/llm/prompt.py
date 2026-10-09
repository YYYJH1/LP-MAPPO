from .vcp_grammar import get_context

FIELD_ORDER = (
    "Write UAV blocks in increasing UAV number, tasks in their visit order. "
    "Each block starts with U<number> layer <A-G>. Each task line is "
    "T<number> g location s a upload priority, in that exact field order. "
    "g is S/G; location is L/P/Q/R; s, a and upload are A-H bins; "
    "priority is X/Y/Z. A local task has upload '-'. "
    "For G write G - - - - Z. Use distinct layers. "
    "Return only these fields, without explanation."
)


def build_prompt(tokenizer, instance_key, registry) -> str:
    from lawn_mec.vcp.serialize import text_serialization

    context = get_context(instance_key, registry)
    content = text_serialization(context.tables, context.dp, context.reference)
    return tokenizer.apply_chat_template(
        [{"role": "system", "content": "Plan UAV service commitments to minimize energy per on-time task. " + FIELD_ORDER},
         {"role": "user", "content": content}],
        tokenize=False, add_generation_prompt=True, enable_thinking=False)


def token_budget(tokenizer, prompt, grammar, max_model_len=4096):
    prefix = ()
    output = 1
    while (field := grammar.next_field(prefix)) is not None:
        output += len(tokenizer.encode(grammar.render_prefix_text(prefix), add_special_tokens=False)) + 1
        prefix += (field.options[0],)
    inputs = len(tokenizer.encode(prompt, add_special_tokens=False))
    return dict(prompt_tokens=inputs, output_tokens_bound=output,
                total_tokens_bound=inputs + output, max_model_len=max_model_len,
                fits=inputs + output <= max_model_len)
