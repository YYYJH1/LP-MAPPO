import numpy as np
import math
from lawn_mec.env.geometry import Box, segment_box
from .tables import finish


def static_features(tables, dp, reference=None):
    free, rush, leave_one = [], [], np.empty(len(tables.tasks))
    for u, own in enumerate(tables.own):
        all_mask = (1 << len(own))-1
        free.append(dp.schedules[u][0].energy)
        rush.append(dp.schedules[u][all_mask].energy)
        for j, k in enumerate(own):
            leave_one[k] = dp.schedules[u][all_mask ^ (1 << j)].energy
    result = dict(free_energy_J=np.asarray(free), all_keep_energy_J=np.asarray(rush),
                  leave_one_out_energy_J=leave_one,
                  visits=np.asarray([[*tables.windows[k, 's'], *tables.windows[k, 'a']] for k in range(len(tables.tasks))]),
                  uploads=np.asarray([tables.upload_windows[k] for k in range(len(tables.tasks))]),
                  readiness=tables.ready_min.copy(), readiness_by_location=tables.ready_by_location.copy(),
                  visit_bins=np.asarray([[tables.visit_bins[k, 's'], tables.visit_bins[k, 'a']] for k in range(len(tables.tasks))]),
                  upload_bins=np.asarray([tables.upload_bins[k] for k in range(len(tables.tasks))]),
                  minimum_travel=tables.min_time.copy())
    result['rush_marginal_J'] = np.array([rush[t['u']]-leave_one[k] if np.isfinite(rush[t['u']]) and np.isfinite(leave_one[k]) else np.nan for k, t in enumerate(tables.tasks)])
    K, M, H = len(tables.tasks), tables.p['M'], len(tables.lib.heights)
    upload = np.empty((K, M, H), dtype=np.int32)
    los = np.zeros_like(upload)
    mec = np.empty((K, M), dtype=np.int32)
    for k, task in enumerate(tables.tasks):
        release = tables.windows[k, 's'][0]+task['L_s']
        for m in range(1, M+1):
            for ell, h in enumerate(tables.lib.heights):
                upload[k, m-1, ell] = tables.node_upload(k, m, ell)
                los[k, m-1, ell] = not any(segment_box(tables.lib.point(task['z_s'], h), tables.instance['bs'][m-1], Box.from_dict(b)) for b in tables.instance['buildings'])
            mec[k, m-1] = finish(task['C'], tables.cpu_capacity[m-1], release, tables.N+1)-release
    result.update(upload_duration=upload, line_of_sight=los, mec_duration=mec,
                  local_duration=np.array([math.ceil(t['C']/(tables.p['delta']*tables.p['F_local'])) for t in tables.tasks]))
    if reference is not None:
        result.update(reference_layers=np.asarray(reference.layers),
                      reference_tasks=np.asarray([[int(c.g == 'S'), -1 if c.location is None else c.location,
                          -1 if c.s is None else c.s, -1 if c.a is None else c.a,
                          -1 if c.upload is None else c.upload, 'XYZ'.index(c.priority)] for c in reference.tasks]))
    return result
