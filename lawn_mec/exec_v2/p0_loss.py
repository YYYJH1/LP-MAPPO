import math

HARD_IDS = tuple(f'C{i:02d}' for i in range(1, 21) if i != 18)


def c_min(K, zeta=.85):
    return math.ceil(zeta*K-1e-9)


def hard_nonpass(audit):
    conditions, fields = audit['conditions'], audit.get('field_conditions') or {}
    failed = {}
    for cid in HARD_IDS:
        if cid == 'C07' and 'viol_visit_continue' in fields:
            status = fields['viol_visit_continue'].get('status')
        else:
            status = conditions.get(cid, {}).get('status')
        if status != 'pass':
            failed[cid] = status
    return failed


def on_time_count(audit):
    return sum(bool(t['completed']) for t in audit['tasks'])


def terminal_loss(audit, instance, zeta=.85):
    K = len(instance['tasks']); minimum = c_min(K, zeta)
    C = on_time_count(audit); hard = hard_nonpass(audit)
    energy = float(sum(u['E_u'] for u in audit['uavs']))
    reference = instance['params']['U']*instance['params']['battery']/minimum
    feasible = C >= minimum and not hard
    loss = energy/C/reference if feasible else 1+max(0, minimum-C)/minimum+len(hard)
    return dict(loss=loss, C=C, K=K, c_min=minimum, on_time=C/K, feasible=feasible,
                upsilon=float(len(hard)), hard=hard, energy_J=energy,
                eta_J=energy/C if C else None, eta_reference_J=reference)


def search_key(audit, minimum):
    C = on_time_count(audit)
    return (len(hard_nonpass(audit)), max(0, minimum-C), -C,
            float(sum(u['E_u'] for u in audit['uavs'])))
