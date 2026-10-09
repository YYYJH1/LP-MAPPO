IDS = tuple(f'C{i:02d}' for i in range(1, 21))
FIELDS = {
    'C01': ['viol_route'], 'C02': ['viol_route'], 'C03': ['viol_route'],
    'C04': ['viol_corridor'], 'C05': ['viol_service_region'], 'C06': ['viol_service_speed'],
    'C07': ['viol_visit_complete', 'viol_visit_continue'], 'C08': ['viol_visit_order'],
    'C09': [], 'C10': ['viol_endpoint'], 'C11': ['viol_speed_flight'], 'C12': ['viol_accel'],
    'C13': ['viol_power_floor'], 'C14': ['viol_separation'], 'C15': ['viol_tdma_bs', 'viol_tdma_uav'],
    'C16': ['viol_ul_power'], 'C17': ['viol_cpu_local', 'viol_cpu_mec'],
    'C18': ['viol_result_ready', 'viol_deadline', 'viol_freshness'], 'C19': ['viol_battery'], 'C20': []}


def verdict(residual, tolerance=0):
    return {'status': 'pass' if residual <= tolerance else 'fail', 'residual': float(residual)}


def merge(*items):
    if not items: return verdict(0)
    return {'status': max((x['status'] for x in items), key={'pass': 0, 'unknown': 1, 'fail': 2}.get),
            'residual': max(x.get('residual', 0.) for x in items)}


def merge_layers(evaluation, audit=None):
    cs = {key: dict(evaluation.get(key, {'status': 'unknown', 'residual': 0})) for key in IDS}
    if audit is not None:
        for key, item in cs.items():
            if item['status'] == 'unknown' and key in audit:
                cs[key] = dict(audit[key])
    return cs


def aggregate(conditions, endpoint_residual=None, epsilon_phys=None, audit_conditions=None):
    cs = merge_layers(conditions, audit_conditions)
    statuses = [x['status'] for x in cs.values()]
    p0 = all(x == 'pass' for x in statuses)
    relaxed = p0
    if endpoint_residual is not None and epsilon_phys is not None:
        relaxed = endpoint_residual <= epsilon_phys and all(x['status'] == 'pass' for k, x in cs.items() if k != 'C10')
    status = 'p0' if p0 else 'relaxed_only' if relaxed else 'infeasible' if 'fail' in statuses else 'unknown'
    return {'feasible_status': status, 'feasible_p0': p0, 'feasible_relaxed': relaxed,
            'checker_unknown_count': statuses.count('unknown')}
