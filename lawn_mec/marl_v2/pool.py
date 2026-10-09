import hashlib
import json
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_manifest(path):
    path = Path(path).resolve()
    data = json.loads(path.read_text())
    if not isinstance(data, dict) or not isinstance(data.get('instances'), list):
        raise ValueError('manifest requires an instances list')
    rows = []; seen = set()
    for original in data['instances']:
        row = dict(original)
        source = Path(row['path'])
        source = source if source.is_absolute() else path.parent/source
        source = source.resolve(); actual = sha(source)
        if row.get('sha256') != actual:
            raise ValueError(f'instance sha256 mismatch: {source}')
        if actual in seen:
            raise ValueError('duplicate instance in pool')
        seen.add(actual); row['path'] = str(source)
        rows.append(row)
    return dict(data, instances=rows, manifest_path=str(path), manifest_sha256=sha(path))


def filter_witnessed(manifest, enabled=True):
    rows = []; discarded = []
    for row in manifest['instances']:
        inst = json.loads(Path(row['path']).read_text())
        cert = inst.get('certification', {})
        reason = None
        if cert.get('status') != 'witness' or cert.get('battery_choice') != 'witness':
            reason = 'no_witness'
        elif not cert.get('witness_hash') or not cert.get('final_replay', {}).get('feasible'):
            reason = 'missing_passing_witness_replay'
        elif cert.get('battery_J') != inst['params']['battery']:
            reason = 'witness_battery_mismatch'
        if reason and enabled:
            discarded.append(dict(row, reason=reason))
        else:
            rows.append(row)
    total = len(manifest['instances'])
    return dict(manifest, instances=rows, filter=dict(enabled=enabled, total=total,
                retained=len(rows), discarded_count=len(discarded), discarded=discarded,
                coverage=len(rows)/total if total else 0.))
