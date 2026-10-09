import numpy as np


def location_guard(features, N):
    rows = np.asarray(features, float)
    hi = np.rint(rows[:, 10]*N).astype(int)
    limit = np.minimum(np.rint(rows[:, 13]*N), np.rint(rows[:, 14]*N)).astype(int)
    return hi <= limit


def stop_loss(r, times, n):
    if r is None or r.certified:
        return None
    times = np.asarray(times)
    if max(r.earliest, r.lo) > r.limit:
        return times == n
    return (times == n) | (times <= r.limit)


def guard_support(env, head, features, mask, request=None):
    mask = np.asarray(mask, bool)
    if not env.flags.f17:
        return mask
    if head == 'location':
        support = mask & location_guard(features, env.N)
        row = env.guard_counts['location']
        row[0] += int(support.any() and not np.array_equal(support, mask))
        row[1] += int(not support.any())
    elif (head == 'timing' and request is not None and request.get('reason') == 'deferral'
          and request.get('visit') is not None and request['visit'][1] == 'a'):
        r = env.readiness(request)
        sl = stop_loss(r, np.rint(np.asarray(features)[:, 0]*env.N).astype(int), env.executor.state.n)
        if sl is None:
            return mask
        support = mask & sl
        row = env.guard_counts['stop_loss']
        row[0] += int(support.any() and not np.array_equal(support, mask))
        row[1] += int(max(r.earliest, r.lo) > r.limit or not support.any())
    else:
        return mask
    return support if support.any() else mask


def guard_summary(counts):
    return dict(location=list(counts['location']), stop_loss=list(counts['stop_loss']),
                guard=sum(v[0] for v in counts.values()))


def merge_stats(summaries):
    out = {}
    for src in summaries:
        for key, value in src.items():
            if isinstance(value, dict):
                out[key] = merge_stats([out.get(key, {}), value])
            elif isinstance(value, list):
                out[key] = [a+b for a, b in zip(out.get(key, [0]*len(value)), value)]
            else:
                out[key] = out.get(key, 0)+value
    return out
