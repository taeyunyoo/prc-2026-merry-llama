"""Content-bound human provenance gate; no model or Parquet dependency."""
import hashlib
import json
from pathlib import Path
from .data import training_names


def file_digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


CONTRACT = 'prc2026.retrospective_backtest.v1'
CHECKS = ('authorized_inputs', 'metadata_train_only', 'external_reference_asof',
          'weather_preparation_reviewed', 'distance_provenance_reviewed',
          'retrospective_batch_features_accepted')


def inventory(paths, relative_to):
    return {p.relative_to(relative_to).as_posix(): file_digest(p) for p in sorted(paths)}


def validate_audit(base, audit_path):
    """Attestations are human evidence, not an automated proof of isolation."""
    audit = json.loads(Path(audit_path).read_text())
    if audit.get('contract') != CONTRACT:
        raise ValueError('Unknown backtest audit contract')
    paths = [base / name for name in training_names()]
    actual = inventory(paths, base)
    if audit.get('base_sha256') != actual:
        raise ValueError('Audit must bind exactly the 12 base datasets by SHA-256')
    for check in CHECKS:
        record = audit.get('checks', {}).get(check, {})
        if (record.get('approved') is not True or not isinstance(record.get('evidence'), str)
                or not record['evidence'].strip()):
            raise ValueError('Unresolved audit requirement: ' + check)
    # This path deliberately makes no selection-independent validation claim.
    if audit.get('model_selection') != 'published_frozen_exploratory':
        raise ValueError('Acknowledge published_frozen_exploratory model selection')
    if not isinstance(audit.get('reviewer'), str) or not audit['reviewer'].strip():
        raise ValueError('Name the audit reviewer')
    return audit
