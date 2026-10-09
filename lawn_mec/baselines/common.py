from functools import lru_cache
import heapq
import math
from pathlib import Path
import numpy as np
import yaml
from lawn_mec.env.geometry import Box
from lawn_mec.env.model import gain, motion


def descent_limit(p):
    if 'v_desc_max' not in p:
        from .reference_schedule import derive_descent_limit
        derived = derive_descent_limit(p)
    else:
        derived = p['v_desc_max']
    return min(p['V_down'], .9*derived)


def config():
    return yaml.safe_load((Path(__file__).parents[1]/'config/algo/baselines.yaml').read_text())


def centers(instance):
    return np.array([Box.from_dict(b).center for b in instance['nodes']])


def node_of(instance, q):
    return next(i for i, b in enumerate(instance['nodes']) if Box.from_dict(b).contains(q, 1e-7))


def _graph_key(instance):
    return (instance['params']['clearance'],
            tuple((tuple(b['lo']), tuple(b['hi'])) for b in instance['nodes']),
            tuple((e['source'], e['target'], tuple(e['box']['lo']), tuple(e['box']['hi']))
                  for e in instance['edges']))


@lru_cache(maxsize=64)
def _eroded_graph(key):
    s, nodes, edges = key
    c = np.array([(np.asarray(lo, float)+np.asarray(hi, float))/2 for lo, hi in nodes])
    graph = {i: [] for i in range(len(c))}
    for idx, (a, b, lo, hi) in enumerate(edges):
        box = Box(lo, hi).erode(s)
        if a != b and box.nonempty and box.contains(c[a]) and box.contains(c[b]):
            graph[a].append((b, float(np.linalg.norm(c[b]-c[a])), idx))
    return tuple(tuple(graph[i]) for i in range(len(c)))


def eroded_graph(instance):
    return {i: list(edges) for i, edges in enumerate(_eroded_graph(_graph_key(instance)))}


@lru_cache(maxsize=4096)
def _dijkstra(key, source, target, banned_edges, banned_nodes):
    graph = _eroded_graph(key); heap = [(0., (source,))]; best = {source: 0.}
    while heap:
        dist, path = heapq.heappop(heap); a = path[-1]
        if dist > best[a]: continue
        if a == target: return path
        for b, w, _ in graph[a]:
            if (a, b) in banned_edges or b in banned_nodes: continue
            value = dist+w
            if value < best.get(b, math.inf):
                best[b] = value; heapq.heappush(heap, (value, (*path, b)))
    raise ValueError(f'no path in eroded graph: {source}->{target}')


def dijkstra(instance, source, target, banned_edges=(), banned_nodes=()):
    return list(_dijkstra(_graph_key(instance), source, target,
                          frozenset(banned_edges), frozenset(banned_nodes)))


def path_length(instance, path):
    c = centers(instance)
    return float(sum(np.linalg.norm(c[b]-c[a]) for a, b in zip(path, path[1:])))


def k_shortest(instance, source, target, k=3):
    accepted = [dijkstra(instance, source, target)]; heap = []; seen = {tuple(accepted[0])}
    for _ in range(1, k):
        previous = accepted[-1]
        for i in range(len(previous)-1):
            root = previous[:i+1]
            banned = {(p[i], p[i+1]) for p in accepted if len(p) > i+1 and p[:i+1] == root}
            try: spur = dijkstra(instance, root[-1], target, banned, root[:-1])
            except ValueError: continue
            candidate = tuple(root[:-1]+spur)
            if candidate not in seen:
                seen.add(candidate); heapq.heappush(heap, (path_length(instance, candidate), candidate))
        if not heap: break
        accepted.append(list(heapq.heappop(heap)[1]))
    return accepted


def line_profile(start, finish, p, scale=1., target_box=None, cfg=None):
    cfg = cfg or config(); start = np.asarray(start, float); d = np.asarray(finish)-start
    distance = np.linalg.norm(d)
    if distance < cfg['position_tolerance']: return []
    if not 0 < scale <= 1: raise ValueError('speed scale must lie in (0,1]')
    h = np.linalg.norm(d[:2]); z = abs(d[2]); dt = p['delta']
    vmax = min(p['V_h']*distance/h if h else math.inf,
               (p['V_up'] if d[2] >= 0 else descent_limit(p))*distance/z if z else math.inf)*scale
    amax = min(p['a_h']*distance/h if h else math.inf, p['a_z']*distance/z if z else math.inf)
    direction = d/distance
    for total in range(2, cfg['profile_search_slots']+1):
        for ramp in range(1, total//2+1):
            cruise = total-2*ramp; speed = distance/(dt*(ramp+cruise)); acceleration = speed/(ramp*dt)
            if speed > vmax+1e-12 or acceleration > amax+1e-12: continue
            acc = [direction*acceleration]*ramp+[np.zeros(3)]*cruise+[-direction*acceleration]*ramp
            if target_box is not None:
                q, v = start.copy(), np.zeros(3); valid = True
                for a in acc:
                    q, v = motion(q, v, a, dt)
                    if target_box.contains(q, 1e-7) and not target_box.erode(p['clearance']).contains(q, 1e-7):
                        valid = False; break
                if not valid: continue
            return [a.copy() for a in acc]
    raise ValueError('profile search budget exhausted')


def profile_samples(instance, path, cfg, scale=1.):
    c = centers(instance); samples = []
    local_cfg={**cfg,'speed_scale':scale}
    for seg in route_segments(instance,path,c[path[0]],c[path[-1]],c[path[0],2],local_cfg):
        q,v=seg['start'].copy(),seg['initial_velocity'].copy()
        for accel in seg['accels']:
            samples.append(motion(q,v,accel,instance['params']['delta']/2)[0])
            q,v=motion(q,v,accel,instance['params']['delta'])
    return samples or [c[path[0]]]


def mean_gains(instance, samples):
    return np.mean([[gain(q, bs, instance['buildings'], instance['params']) for bs in instance['bs']] for q in samples], axis=0)


def choose_path(instance, source, target, algo, cfg):
    paths = k_shortest(instance, source, target, cfg['path_candidates'] if algo == 'B3' else 1)
    lengths = np.array([path_length(instance, path) for path in paths])
    gains = np.array([max(mean_gains(instance, profile_samples(instance, path, cfg))) for path in paths])
    scores = lengths/max(float(max(lengths)), 1.)-cfg['path_gain_weight']*gains/max(float(max(gains)), np.finfo(float).tiny)
    selected = int(np.argmin(scores))
    return paths[selected], {'paths': paths, 'lengths': lengths.tolist(), 'best_mean_gains': gains.tolist(), 'scores': scores.tolist(), 'selected': selected}


def window_upper(task, state, dt):
    ns = state.events['s'].value
    return min(task['d']-task['L_a']*dt, ns*dt+task['H'] if ns is not None else math.inf)


def min_power(work, gains, sigma, p, cfg):
    gains = np.asarray(gains, float); sigma = np.asarray(sigma, float)
    def capacity(power):
        return float(np.sum(sigma*p['delta']*p['bandwidth']*np.log2(1+power*gains/(p['N0']*p['bandwidth']))))
    if work <= 0: return 0., True
    cap = p['P_ul']
    if capacity(cap) < work: return cap, False
    lo, hi = 0., cap
    for _ in range(cfg['power_bisections']):
        mid = (lo+hi)/2
        if capacity(mid) >= work: hi = mid
        else: lo = mid
    return min(cap, hi*(1+cfg['power_margin'])), True


def just_enough_cpu(env, workloads, locations):
    shares = {}
    for u in range(env.p['U']):
        remaining = env.p['F_local']
        ks = [k for k, t in enumerate(env.tasks) if t['u'] == u and locations[k] == 0 and workloads[k][1] > 0]
        for k in sorted(ks, key=lambda k: (window_upper(env.tasks[k], env.states[k], env.p['delta']), k)):
            f = min(workloads[k][1]/env.p['delta'], remaining); remaining -= f
            shares[k] = f/env.p['F_local']
    return shares


def layer_heights(instance):
    p = instance['params']; low = p['h_min']+p['clearance']; high = p['h_max']-p['clearance']
    for t in instance['tasks']:
        for phase in ('s', 'a'):
            box = Box.from_dict(t['S_'+phase])
            low, high = max(low, box.lo[2]), min(high, box.hi[2])
    heights = (low+high)/2+(np.arange(p['U'])-(p['U']-1)/2)*(p['D_z']+1)
    if np.any(heights < low) or np.any(heights > high):
        raise ValueError(f'height_layers_do_not_fit: U={p["U"]}, band=[{low},{high}]')
    return heights


def merged_path(instance, path, height):
    points = centers(instance)[path].copy(); points[:, 2] = height
    keep = [0]
    for i in range(1, len(points)-1):
        a, b = points[i]-points[i-1], points[i+1]-points[i]
        if not np.allclose(a/np.linalg.norm(a), b/np.linalg.norm(b)): keep.append(i)
    if len(points)>1: keep.append(len(points)-1)
    return [(path[i], points[i]) for i in keep]


def moving_line(start, finish, p, cfg, initial=0., final=0., scale=1., vmax=None, nodes=()):
    start, finish = np.asarray(start), np.asarray(finish)
    d = finish-start; length = float(np.linalg.norm(d)); dt = p['delta']
    if length < cfg['position_tolerance']: return []
    direction = d/length
    horizontal = np.linalg.norm(direction[:2]); vertical = abs(direction[2])
    limit = min(p['V_h']/horizontal if horizontal else math.inf,
                (p['V_up'] if d[2]>0 else descent_limit(p))/vertical if vertical else math.inf)*scale
    if vmax is not None: limit = min(limit, vmax)
    amax = min(p['a_h']/horizontal if horizontal else math.inf,
               p['a_z']/vertical if vertical else math.inf)
    if max(initial, final) > limit+1e-9: raise ValueError('boundary speed exceeds line limit')
    for total in range(1, min(cfg['profile_search_slots'], int(2*length/max(limit,1e-6)/dt+30))+1):
        for up in range(total+1):
            for down in range(total-up+1):
                cruise = total-up-down; denom = cruise+(up+down)/2
                if not denom: continue
                peak = (length/dt-(up*initial+down*final)/2)/denom
                if peak < -1e-10 or peak > limit+1e-10: continue
                if (up==0 and abs(peak-initial)>1e-9) or (down==0 and abs(peak-final)>1e-9): continue
                au = (peak-initial)/(up*dt) if up else 0.
                ad = (peak-final)/(down*dt) if down else 0.
                if max(abs(au),abs(ad))>amax+1e-10: continue
                scalars = [au]*up+[0.]*cruise+[-ad]*down
                velocities = initial+dt*np.cumsum([0.]+scalars)
                distances = dt*np.cumsum((velocities[:-1]+velocities[1:])/2)
                qs = start+distances[:,None]*direction
                if any(np.any(np.all((qs >= box.lo-1e-8)&(qs <= box.hi+1e-8),axis=1)
                                  & ~np.all((qs >= box.lo+p['clearance']-1e-8)&(qs <= box.hi-p['clearance']+1e-8),axis=1))
                       for box in nodes): continue
                return [direction*a for a in scalars]
    raise ValueError('moving profile search budget exhausted')


def route_segments(instance, path, start, finish, height, cfg, scales=None, service=None, service_limit=None):
    p = instance['params']; dt = p['delta']; points = merged_path(instance,path,height)
    segments=[]; q=np.asarray(start,float).copy()
    boxes=[Box.from_dict(instance['nodes'][z]) for z in path[1:]]
    def line(end, initial=0., final=0., vmax=None, kind='straight'):
        nonlocal q
        if np.linalg.norm(end-q) < cfg['position_tolerance']: return
        scale=cfg['speed_scale'] if scales is None else scales[len(segments)]
        aa=moving_line(q,end,p,cfg,initial,final,scale,vmax,boxes)
        segments.append({'finish':np.asarray(end).copy(),'accels':aa,'kind':kind,'start':q.copy(),
                         'initial_velocity':((np.asarray(end)-q)/np.linalg.norm(end-q)*initial)})
        q=np.asarray(end).copy()
    line(points[0][1],kind='in_node')
    speed=0.
    for i in range(1,len(points)):
        center=points[i][1]; incoming=(center-points[i-1][1]); incoming/=np.linalg.norm(incoming)
        if i<len(points)-1:
            outgoing=points[i+1][1]-center; outgoing/=np.linalg.norm(outgoing)
            vc=min(cfg['corner_speed'],p['a_h']*dt/np.linalg.norm(outgoing-incoming))
            b=Box.from_dict(instance['nodes'][points[i][0]]).erode(p['clearance'])
            vc=min(vc,2*float(np.min(np.minimum(center-b.lo,b.hi-center)))/dt)
            if scales is not None: vc=min(vc,p['V_h']*min(scales))
            entry=center-incoming*vc*dt/2
            line(entry,speed,vc)
            accel=(outgoing-incoming)*vc/dt
            end=center+outgoing*vc*dt/2
            segments.append({'finish':end,'accels':[accel],'kind':'turn','start':entry,
                             'initial_velocity':incoming*vc,'corner_speed':vc})
            q=end; speed=vc
        else:
            if service is not None:
                b=Box.from_dict(service)
                axis=int(np.argmax(np.abs(incoming)))
                boundary=center-incoming*((Box.from_dict(instance['nodes'][path[-1]]).hi[axis]-Box.from_dict(instance['nodes'][path[-1]]).lo[axis])/2-p['clearance']-1e-4)
                approach=min(service_limit if service_limit is not None else p['V_service'],p['V_h']*(min(scales) if scales else cfg['speed_scale']))
                line(boundary,speed,approach)
                line(center,approach,0.,approach,kind='service_approach')
            else: line(center,speed,0.)
            speed=0.
    line(np.asarray(finish),kind='in_node')
    return segments
