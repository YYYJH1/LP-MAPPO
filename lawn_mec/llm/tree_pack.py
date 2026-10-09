from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class Block:
    parent: int
    tokens: tuple[int, ...]
    position: int


class TreePacking:
    def __init__(self, tree, tokenizer):
        self.tree = tree
        self.prompt = tuple(tokenizer.encode(tree.prompt, add_special_tokens=False))
        if not self.prompt or not tree.nodes or tree.nodes[0].index != 0 or tree.nodes[0].sample is not None:
            raise ValueError('packing requires a nonempty prompt and an empty root at index 0')
        self.blocks = [Block(-1, self.prompt, 0)]
        self.paths = {0: ()}
        unique = {}
        for index, node in enumerate(tree.nodes[1:], 1):
            if node.index != index or node.parent not in self.paths or node.sample is None:
                raise ValueError('tree nodes must be indexed in parent-before-child order')
            trace = node.sample.trace
            parent = tree.nodes[node.parent]
            before = self.prompt if parent.sample is None else parent.sample.trace.token_ids
            fields = () if parent.sample is None else parent.sample.trace.fields
            prefix = () if parent.sample is None else parent.sample.trace.prefix
            start = node.sample.start_field
            if (trace.prompt != tree.prompt or trace.instance_key != tree.instance_key or
                    node.depth != parent.depth + 1 or start != len(fields) or
                    trace.fields[:start] != fields or trace.prefix[:start] != prefix or
                    trace.token_ids[:len(before)] != before or len(trace.token_ids) <= len(before) or
                    len(trace.fields) <= start):
                raise ValueError('trace does not extend its tree parent')
            for f in trace.fields:
                if (not 0 < f.position < len(trace.token_ids) or not f.allowed_ids or
                        not 0 <= f.selected < len(f.allowed_ids) or
                        trace.token_ids[f.position] != f.allowed_ids[f.selected]):
                    raise ValueError('invalid field position or selected token')
            if trace.fields[start].position < len(before):
                raise ValueError('new block fields overlap the parent')
            path = self.paths[node.parent]
            parent_block = path[-1] if path else 0
            suffix = trace.token_ids[len(before):]
            key = (parent_block, suffix)
            if key not in unique:
                unique[key] = len(self.blocks)
                self.blocks.append(Block(parent_block, suffix, len(before)))
            self.paths[index] = (*path, unique[key])

    def chunks(self, max_blocks=32):
        if not isinstance(max_blocks, int) or isinstance(max_blocks, bool) or max_blocks < 1:
            raise ValueError('max_blocks must be a positive integer')
        if max(map(len, self.paths.values())) > max_blocks:
            raise ValueError('max_blocks must fit the deepest ancestor path')
        nodes, active = [], set()
        for node in self.tree.nodes[1:]:
            required = set(self.paths[node.index])
            if nodes and len(active | required) > max_blocks:
                yield nodes
                nodes, active = [], set()
            nodes.append(node)
            active.update(required)
        if nodes:
            yield nodes

    def layout(self, nodes):
        active = sorted({block for node in nodes for block in self.paths[node.index]})
        tokens, positions = list(self.prompt), list(range(len(self.prompt)))
        ranges = {0: (0, len(self.prompt))}
        for index in active:
            block = self.blocks[index]
            start = len(tokens)
            tokens.extend(block.tokens)
            positions.extend(range(block.position, block.position + len(block.tokens)))
            ranges[index] = (start, len(tokens))
        paths = {}
        for node in nodes:
            paths[node.index] = tuple(i for block in (0, *self.paths[node.index])
                                      for i in range(*ranges[block]))
        return tokens, positions, ranges, paths

    def arguments(self, nodes, *, device, dtype):
        tokens, positions, ranges, paths = self.layout(nodes)
        mask = torch.full((len(tokens), len(tokens)), torch.finfo(dtype).min, device=device, dtype=dtype)
        for index, (start, end) in ranges.items():
            mask[start:end, start:end] = torch.triu(
                mask.new_full((end-start, end-start), torch.finfo(dtype).min), diagonal=1)
            parent = self.blocks[index].parent
            while parent >= 0:
                lo, hi = ranges[parent]
                mask[start:end, lo:hi] = 0
                parent = self.blocks[parent].parent
        arguments = dict(input_ids=torch.tensor([tokens], dtype=torch.long, device=device),
                         attention_mask=mask[None, None],
                         position_ids=torch.tensor([positions], dtype=torch.long, device=device), use_cache=False)
        return arguments, paths
