from copy import deepcopy
import math


def summary_audit(audit):
    from .search import objective, feasible, certified_energies
    K = len(audit['tasks']); C = sum(t['completed'] for t in audit['tasks'])
    energies = [u['E_u'] for u in audit['uavs']]
    return dict(C=C, K=K, on_time=C/K, energies_J=energies, energy_J=sum(energies),
                eta_J=sum(energies)/C if C else None, feasible=feasible(audit),
                objective=list(objective(audit, math.ceil(.85*K-1e-9))),
                energies_upper_J=certified_energies(audit),
                conditions=audit['conditions'], tasks=audit['tasks'],
                hard_nonpass={k: v for k, v in audit['conditions'].items() if k != 'C18' and v['status'] != 'pass'},
                cpu_energy_J=sum(u['E_c'] for u in audit['uavs']),
                radio_energy_J=sum(u['E_r'] for u in audit['uavs']))


def anchor_instance(instance, result, margin):
    from lawn_mec.env.checker import replay_episode
    from lawn_mec.env.generator import free_flight_anchor
    from lawn_mec.eval.oracle import necessary_bounds
    from .search import feasible, digest, VERSION
    if not math.isfinite(margin) or margin < 0:
        raise ValueError('finite nonnegative margin required')
    final = deepcopy(instance)
    record = dict(status='none', planner_version=VERSION, battery_margin=margin,
                  placeholder_battery_J=instance['params']['battery'], budget_s=result['budget_s'],
                  budget_used_cpu_s=result['cpu_s'], energy_definition='outward_audit_upper_including_subtraction_rounding')
    best = result['incumbent']
    if result['status'] == 'witness':
        final['params']['battery'] = (1+margin)*max(best['energies_upper_J'])
        audit = replay_episode(final, best['result']['logs'], 'audit')
        if not feasible(audit):
            raise AssertionError('battery-anchored witness failed independent frozen audit')
        record.update(status='witness', battery_choice='witness', witness_hash=digest(best['result']['logs']),
                      energies_J=best['energies_upper_J'], final_replay=summary_audit(audit))
    else:
        diagnostic = deepcopy(instance)
        diagnostic['params'].update(battery_margin=margin, battery_aggregate='max')
        free = free_flight_anchor(diagnostic)
        final['params']['battery'] = diagnostic['params']['battery']
        record.update(battery_choice='free_flight_fallback', free_flight=free,
                      incumbent_C=best['result']['C'] if best else None)
    final['params'].update(battery_anchor='witness', battery_margin=margin, battery_aggregate='max')
    record['battery_J'] = final['params']['battery']
    final['certification'] = record
    final['battery_anchor'] = dict(kind=record['battery_choice'], margin=margin, aggregate='max',
                                   E_witness_u=record.get('energies_J'), battery_J=record['battery_J'])
    final['oracle'] = necessary_bounds(final)
    final['state'] = 'feasible_witness' if record['status'] == 'witness' else 'undetermined'
    return final


