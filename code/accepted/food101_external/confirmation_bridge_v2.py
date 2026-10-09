"""One frozen Food101 revision confirmation. Build all states before scoring.

Exact approved current_core and rectangular Procrustes are imported unchanged.
Only input dimensions and already-frozen metadata manifests are generalized.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

ROOT = Path(__file__).resolve().parents[1]
PROTO = ROOT / 'provenance/protocol_freeze'
STREAMS = {'A': 2, 'B': 3, 'C': 4}
STAGES = ('before_R1', 'after_refresh_R1', 'after_R1', 'before_R2', 'after_refresh_R2', 'after_R2')
METHODS = ('mean', 'paired025', 'sample_current', 'full_target_ridge')
CORE_SHA = '0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7'
PROCRUSTES_SHA = '59be9742d62f67aa6b1c3c47e139932fa4686f1916a8cfc8b8c2c52c353bb78f'


def utc():
    return datetime.now(timezone.utc).isoformat()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def identity(path):
    path = Path(path).resolve()
    return {'path': str(path), 'bytes': path.stat().st_size, 'sha256': sha(path)}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('utf-8')


def digest(value):
    return hashlib.sha256(canonical(value)).hexdigest()


def semantic(values):
    records = {k: {'shape': list(a.shape), 'dtype': a.dtype.str,
                   'c_order_raw_sha256': hashlib.sha256(np.ascontiguousarray(a).tobytes(order='C')).hexdigest()}
               for k, a in sorted(values.items())}
    return records, digest(records)


def array_identity(path):
    with np.load(path, allow_pickle=False) as saved:
        records, sd = semantic({k: saved[k] for k in saved.files})
    return identity(path) | {'arrays': records, 'semantic_digest': sd}


def write(path, data):
    with Path(path).open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n')


def save(path, arrays):
    with Path(path).open('xb') as stream:
        np.savez_compressed(stream, **arrays)
    return array_identity(path)


def csvwrite(path, rows):
    if not rows:
        raise ValueError('Empty required CSV')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


class AccessBoundary:
    """Reject images/models/network/processes; isolate fold5 NPZ reads by role."""
    def __init__(self, phase):
        self.phase, self.role = phase, 'bootstrap'
        self.forbidden = []
        self.housekeeping_metadata_denials = []
        self.feature_reads = []
        self.real_image_opens = 0

    def metadata_denial(self, event, detail):
        # Still deny the actual operation. Classify only exact import-time callers.
        if self.role != 'numerical_library_bootstrap':
            return False
        stack = []
        frame = sys._getframe(1)
        while frame is not None:
            stack.append({'file': frame.f_code.co_filename, 'line': frame.f_lineno,
                          'function': frame.f_code.co_name})
            frame = frame.f_back
        stack.reverse()
        utils = str(ROOT.parent / '.venv/Lib/site-packages/numpy/testing/_private/utils.py')
        stdlib = str(Path(sys.base_prefix) / 'Lib')
        norm = lambda p: str(Path(p)).casefold()
        def has(path, function, line=None):
            return any(norm(x['file']) == norm(path) and x['function'] == function
                       and (line is None or x['line'] == line) for x in stack)
        if not (has(utils, '<module>', 90) and has(stdlib + '/platform.py', 'machine')
                and has(stdlib + '/platform.py', 'uname')):
            return False
        expected = ((event == 'socket.gethostname' and detail == '()'
                     and has(stdlib + '/platform.py', '_node'))
                    or (event == 'open' and detail == 'write outside new task root: ' + r'\\.\NUL'
                        and has(stdlib + '/platform.py', '_syscmd_ver')
                        and has(stdlib + '/subprocess.py', '_get_devnull')))
        if not expected:
            return False
        self.housekeeping_metadata_denials.append({'event': event, 'detail': detail,
            'role': self.role, 'source_stack': stack, 'actual_operation_executed': False})
        raise PermissionError('Denied exact numerical-library metadata probe: ' + detail)

    def reject(self, event, detail):
        self.metadata_denial(event, str(detail))
        self.forbidden.append({'event': event, 'detail': str(detail), 'role': self.role})
        raise PermissionError('Forbidden confirmation access: ' + str(detail))

    def audit(self, event, args):
        if event.startswith(('socket.', 'subprocess.', 'os.system', 'os.exec', 'os.spawn')):
            self.reject(event, args[:2])
        if event != 'open' or not args or isinstance(args[0], int):
            return
        try:
            path = Path(os.fsdecode(args[0])).resolve()
        except (TypeError, ValueError):
            return
        suffix = path.suffix.lower()
        mode = str(args[1] or '') if len(args) > 1 else ''
        flags = args[2] if len(args) > 2 and isinstance(args[2], int) else 0
        writing = any(x in mode for x in 'wax+') or bool(flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and not path.is_relative_to(ROOT):
            self.reject(event, 'write outside new task root: ' + str(path))
        if suffix in ('.jpg', '.jpeg', '.png', '.ppm', '.pth', '.pt'):
            self.reject(event, 'image or model payload: ' + str(path))
        if suffix in ('.npz', '.npy') and not writing:
            if not path.is_relative_to(ROOT):
                self.reject(event, 'outside confirmation numeric assets: ' + str(path))
            if path.is_relative_to(ROOT / 'features'):
                if path.parent.name not in ('resnet18', 'swin_t') or path.stem not in ('fold2', 'fold3', 'fold4', 'fold5'):
                    self.reject(event, 'unfrozen fold/encoder numeric asset')
                if path.stem == 'fold5' and self.role not in ('cache_integrity', 'evaluation'):
                    self.reject(event, 'fold5 outside integrity/evaluation')
                self.feature_reads.append({'path': str(path), 'role': self.role, 'utc': utc()})

    def assert_clean(self):
        if self.phase == 'build':
            events = [x['event'] for x in self.housekeeping_metadata_denials]
            if events.count('socket.gethostname') != 1 or events.count('open') != 3 or len(events) != 4:
                raise RuntimeError('Metadata denial pattern differs from preserved diagnosis; STOP')
        if self.forbidden or any(k == 'torch' or k.startswith('torchvision') for k in sys.modules):
            raise RuntimeError('Forbidden access or model import; STOP')


BOUNDARY = None


def role(name):
    BOUNDARY.role = name


def verify_sources():
    for name in ('EXECUTION_SOURCE_LOCK.json', 'AUDIT_SOURCE_LOCK.json', 'EXECUTION_SOURCE_LOCK_V2.json', 'AUDIT_SOURCE_LOCK_V2.json'):
        lock = read(ROOT / name)
        if lock['status'] != 'FROZEN_BEFORE_DEPLOYMENT_FEATURE_OR_CONFIRMATION_EXECUTION':
            raise RuntimeError('Execution source not frozen')
        for rel, item in lock['files'].items():
            p = ROOT / rel
            if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
                raise RuntimeError('Source/provenance changed: ' + rel)
    for name in ('PROTOCOL_GATE.json', 'QUALIFICATION_GATE.json', 'PROCRUSTES_SOURCE_GATE.json', 'RUNTIME_GATE.json'):
        if read(ROOT / name)['status'] != 'PASS':
            raise RuntimeError('Gate did not PASS: ' + name)
    if read(ROOT / 'WEIGHT_IDENTITY.json')['status'] != 'ALL_TWO_WEIGHT_IDENTITIES_PASS':
        raise RuntimeError('Two exact official weight identities not PASS')
    if sha(ROOT / 'code/current_core.py') != CORE_SHA or sha(ROOT / 'code/procrustes_alignment.py') != PROCRUSTES_SHA:
        raise RuntimeError('Exact accepted mathematical source changed')
    contract = read(ROOT / 'ARTIFACT_CONTRACT.json')
    if contract['alpha'] != .25 or contract['ridge'] != 1 or contract['shrink'] != .1 or contract['classes'] != 101:
        raise RuntimeError('Frozen method changed')
    engineering = read(ROOT / 'audit/BUILD_GUARD_ENGINEERING_REPAIR.json')
    for path, item in engineering['exact_dependency_source_identities'].items():
        if Path(path).stat().st_size != item['bytes'] or sha(Path(path)) != item['sha256']:
            raise RuntimeError('Exact metadata dependency source changed')
    return contract


def verify_all_features():
    role('cache_integrity')
    lock = read(ROOT / 'ENCODER_FEATURE_LOCK.json')
    if lock['status'] != 'ALL_EIGHT_DEPLOYMENT_FEATURE_CACHES_FROZEN_BEFORE_FRONTEND_OR_PERFORMANCE':
        raise RuntimeError('All-eight deployment cache gate not PASS')
    expected = {f'{enc}_fold{fold}' for enc in ('resnet18', 'swin_t') for fold in range(2, 6)}
    if set(lock['caches']) != expected:
        raise RuntimeError('Feature cache set differs')
    data = read(ROOT / 'DEPLOYMENT_DATA_LOCK.json')
    if data['status'] != 'DEPLOYMENT_DATA_LOCK_PASS':
        raise RuntimeError('Frozen deployment data identity not PASS')
    for encoder in ('resnet18', 'swin_t'):
        for fold in range(2, 6):
            path = ROOT / f'features/{encoder}/fold{fold}.npz'
            cached = lock['caches'][f'{encoder}_fold{fold}']
            expected_identity = cached.get('identity', cached)
            actual = array_identity(path)
            for k in ('bytes', 'sha256', 'semantic_digest'):
                if actual[k] != expected_identity[k]:
                    raise RuntimeError('Feature identity differs: ' + str(path))
            group = data['folds'][f'fold{fold}']
            with np.load(path, allow_pickle=False) as values:
                if set(values.files) != {'features', 'image_ids', 'labels', 'rows'}:
                    raise RuntimeError('Feature keys differ')
                matrix = values['features']
                if matrix.shape != (len(group['images']), 512 if encoder == 'resnet18' else 768) or matrix.dtype.str != '<f4':
                    raise RuntimeError('Feature shape/dtype differs')
                if not np.isfinite(matrix).all() or np.max(np.abs(np.linalg.norm(matrix.astype(np.float64), axis=1) - 1)) > 2e-6:
                    raise RuntimeError('Feature finite/row-L2 gate failed')
                for key, raw in (('image_ids', [r['image_id'] for r in group['images']]), ('labels', [r['class_id'] for r in group['images']]), ('rows', group['rows'])):
                    if values[key].tolist() != raw:
                        raise RuntimeError('Feature canonical discrete identity differs')
    role('build' if BOUNDARY.phase == 'build' else 'evaluation')
    return lock, data


def aug(x):
    return np.column_stack((np.asarray(x, dtype=np.float64), np.ones(len(x))))


def load_numeric(path):
    with np.load(path, allow_pickle=False) as values:
        return {k: values[k].copy() for k in values.files}


class TargetBank:
    """Physical stream cache; candidate gets only manifest-gated exact rows."""
    def __init__(self, stream, fold, group):
        if fold not in (2, 3, 4):
            raise ValueError('Evaluator/qualification fold cannot be candidate bank')
        self.stream, self.fold = stream, fold
        self.group = group
        self.index = {int(r): i for i, r in enumerate(group['rows'])}
        self.ids = {int(r['row']): r['image_id'] for r in group['images']}
        self._target = load_numeric(ROOT / f'features/swin_t/fold{fold}.npz')['features']
        self.calls = []

    def fetch(self, rows, permitted, purpose):
        rows = list(map(int, rows))
        if rows != list(map(int, permitted)) or len(set(rows)) != len(rows):
            raise ValueError('Exact target request differs from frozen receipt')
        role(purpose)
        values = self._target[[self.index[r] for r in rows]].astype(np.float64)
        self.calls.append({'role': purpose, 'global_rows': rows, 'image_ids': [self.ids[r] for r in rows],
                           'bytes': int(values.nbytes), 'c_order_raw_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest(), 'utc': utc()})
        return values

    def full_reference(self, rows):
        role('full_reference_only')
        rows = list(map(int, rows))
        values = aug(self._target[[self.index[r] for r in rows]])
        self.calls.append({'role': 'full_reference_only', 'rows': len(rows), 'candidate_acquisition': False, 'utc': utc()})
        return values


def build():
    started, clock = utc(), time.perf_counter()
    verify_sources()
    feature_lock, data = verify_all_features()
    if started <= feature_lock['created_utc']:
        raise RuntimeError('Frontend build began before deployment feature freeze')
    role('numerical_library_bootstrap')
    from current_core import CurrentEvidenceState, LabelChange, cross, solve
    from procrustes_alignment import fit_rectangular_procrustes, apply_procrustes
    role('build')
    if read(ROOT / 'QUALIFICATION_GATE.json').get('selected_encoder', 'swin_t') != 'swin_t':
        raise RuntimeError('Selected encoder differs')
    receipts, files, frontend_locks, costs, request_export = {}, {}, {}, [], []
    for stream, fold in STREAMS.items():
        out = ROOT / f'states/{stream}'
        out.mkdir(exist_ok=False)
        stage_lock = read(PROTO / f'STREAM_STAGE_LOCKS/{stream}.json')
        evidence = read(PROTO / f'EVIDENCE_LOCKS/{stream}.json')
        corruption = read(PROTO / f'CORRUPTION_LOCKS/{stream}.json')
        request = read(PROTO / f'REQUEST_LOCKS/{stream}.json')
        history = stage_lock['groups']['history']['rows']
        arrivals = stage_lock['groups']['E']['rows']
        fit_global, pilot_global = (evidence['groups'][x]['rows'] for x in ('fit', 'pilot'))
        state_rows = list(history) + list(arrivals)
        idx = {int(r): i for i, r in enumerate(state_rows)}
        if len(fit_global) != 512 or len(pilot_global) != 512 or set(fit_global) & set(pilot_global):
            raise RuntimeError('Frozen fit/pilot identity invalid')
        bank = TargetBank(stream, fold, data['folds'][f'fold{fold}'])
        old = load_numeric(ROOT / f'features/resnet18/fold{fold}.npz')['features']
        old_index = {int(r): i for i, r in enumerate(data['folds'][f'fold{fold}']['rows'])}
        old_history = old[[old_index[r] for r in history]]
        fit, pilot = (np.array([idx[r] for r in group], dtype=np.int64) for group in (fit_global, pilot_global))
        fit_target = bank.fetch(fit_global, fit_global, 'frontend_fit_only')
        pilot_target = bank.fetch(pilot_global, pilot_global, 'frontend_pilot_only')
        timer = time.perf_counter()
        mapping = fit_rectangular_procrustes(old_history[fit], fit_target, mode='similarity', old_pilot=old_history[pilot], new_pilot=pilot_target)
        fit_seconds = time.perf_counter() - timer
        timer = time.perf_counter()
        proxy = aug(apply_procrustes(mapping, old_history))
        map_seconds = time.perf_counter() - timer
        mp = save(out / 'PROCRUSTES_PARAMETERS.npz', {'old_mean': mapping.old_mean, 'new_mean': mapping.new_mean,
                  'rotation': mapping.rotation, 'scale': np.asarray(mapping.scale), 'pilot_bias': mapping.pilot_bias})
        pp = save(out / 'PROXIES.npz', {'history_proxy': proxy, 'global_rows': np.asarray(history, dtype=np.int64),
                  'image_ids': np.asarray([bank.ids[int(r)] for r in history], dtype='<U64')})
        front = {'status': 'EXACT_ACCEPTED_FRONTEND_FROZEN', 'stream': stream, 'source_fold': fold,
                 'old_dim': 512, 'new_dim': 768, 'mode': 'similarity', 'fit_global_rows': fit_global,
                 'pilot_global_rows': pilot_global, 'fit_state_rows': fit.tolist(), 'pilot_state_rows': pilot.tolist(),
                 'fit_image_ids': [bank.ids[r] for r in fit_global], 'pilot_image_ids': [bank.ids[r] for r in pilot_global],
                 'procrustes_source_sha256': PROCRUSTES_SHA, 'source_gate_identity': identity(ROOT / 'PROCRUSTES_SOURCE_GATE.json'),
                 'parameters': mp, 'history_proxy': pp, 'post_normalization': False, 'pilot_refit': False,
                 'fit_seconds': fit_seconds, 'history_map_seconds': map_seconds, 'frozen_utc': utc()}
        write(ROOT / f'FRONTEND_LOCKS/{stream}.json', front)
        frontend_locks[stream] = identity(ROOT / f'FRONTEND_LOCKS/{stream}.json')
        labels_by_row = {int(x['row']): x for x in corruption['images']}
        labels = np.asarray([labels_by_row[int(r)]['historical_label'] for r in history], dtype=np.int64)
        known = np.zeros(len(history), dtype=bool)
        known[np.r_[fit, pilot]] = True
        observed = np.zeros_like(proxy)
        observed[fit], observed[pilot] = aug(fit_target), aug(pilot_target)
        certainty = np.zeros(len(history), dtype=bool)
        certainty[fit] = True
        role('candidate_initial_common_evidence')
        timer = time.perf_counter()
        state = CurrentEvidenceState(proxy, labels, known, observed, certainty, pilot, 101, ridge=1., shrink=.1, feature_version='v2')
        costs.append({'stream': stream, 'operation': 'initial_current_state', 'seconds': time.perf_counter() - timer})
        initial = sorted(list(fit_global) + list(pilot_global))
        acquired = set(initial)
        receipt = {'shared_methods': ['mean', 'paired025', 'sample_current'], 'physical_cache_not_candidate_evidence': True,
                   'initial': {'global_rows': initial, 'image_ids': [bank.ids[r] for r in initial], 'unique_rows': 1024,
                               'known_state_rows': np.flatnonzero(state.known).tolist(), 'certainty_state_rows': fit.tolist()},
                   'events': {}, 'stages': {}}
        discrete = {}

        def capture(stage):
            capture_clock = time.perf_counter()
            n = len(state.labels)
            globals_now = state_rows[:n]
            role('candidate_state_capture')
            arrays = {k: np.asarray(getattr(state, k)).copy() for k in ('proxy', 'observed', 'means', 'known', 'certainty', 'pilot', 'labels', 'versions', 'sample_weights', 'q_mean', 'h_mean', 'q_sample', 'h_sample', 'w_mean', 'w_sample')}
            arrays['w_paired025'] = state.weights(.25)
            arrays['global_rows'] = np.asarray(globals_now, dtype=np.int64)
            arrays['image_ids'] = np.asarray([bank.ids[int(r)] for r in globals_now], dtype='<U64')
            # Evaluator/reference-only compartment; no candidate receives this matrix.
            reference = bank.full_reference(globals_now)
            reference_clock = time.perf_counter()
            arrays['w_full_target_ridge'] = solve(reference.T @ reference + np.eye(reference.shape[1]), cross(reference, state.labels, 101))
            reference_seconds = time.perf_counter() - reference_clock
            role('candidate_state_export')
            file_id = save(out / f'{stage}.npz', arrays)
            files[f'states/{stream}/{stage}.npz'] = file_id
            evidence_arrays = {k: arrays[k] for k in ('proxy', 'observed', 'known', 'certainty', 'pilot', 'labels', 'versions', 'global_rows', 'image_ids')}
            _, evidence_sha = semantic(evidence_arrays)
            receipt['stages'][stage] = {'shared_receipt_sha256': evidence_sha, 'mean_receipt_sha256': evidence_sha,
                                       'paired025_receipt_sha256': evidence_sha, 'sample_current_receipt_sha256': evidence_sha,
                                       'captured_utc': utc(),
                                       'known_rows': int(state.known.sum()), 'certainty_rows': int(state.certainty.sum()),
                                       'unique_candidate_target_global_rows': sorted(acquired), 'file_identity': file_id}
            discrete[stage] = {k: arrays[k].tolist() for k in ('labels', 'versions', 'known', 'certainty', 'pilot', 'global_rows', 'image_ids')}
            costs.append({'stream': stream, 'operation': 'state_' + stage, 'seconds': time.perf_counter() - capture_clock,
                          'full_reference_solve_seconds': reference_seconds, 'candidate_state_array_bytes': sum(state.numeric_payload().values()),
                          'reference_weights_bytes': int(arrays['w_full_target_ridge'].nbytes), 'fold5_scores_computed': False})

        capture('before_R1')
        for event in ('R1', 'R2'):
            if event == 'R2':
                appended = aug(bank.fetch(arrivals, arrivals, 'candidate_E_append'))
                arrival_labels = np.asarray([labels_by_row[int(r)]['historical_label'] for r in arrivals], dtype=np.int64)
                timer = time.perf_counter()
                state.append(appended, arrival_labels, feature_version='v2')
                costs.append({'stream': stream, 'operation': 'append_E', 'seconds': time.perf_counter() - timer})
                acquired.update(arrivals)
                receipt['events']['E_append'] = {'global_rows': list(arrivals), 'image_ids': [bank.ids[r] for r in arrivals],
                                                  'unique_rows': len(arrivals), 'historical_labels': arrival_labels.tolist()}
                capture('before_R2')
            requested = request['requests'][event]['images']
            announced_global = [int(x['row']) for x in requested]
            rows = np.asarray([idx[r] for r in announced_global], dtype=np.int64)
            if np.any(rows >= len(state.labels)):
                raise RuntimeError('Unarrived row requested')
            fresh_global = [r for r, k in zip(announced_global, state.known[rows]) if not k]
            already_global = [r for r, k in zip(announced_global, state.known[rows]) if k]
            values = aug(bank.fetch(announced_global, announced_global, 'candidate_' + event + '_refresh'))
            timer = time.perf_counter()
            state.observe(rows, values, feature_version='v2', authoritative=True)
            costs.append({'stream': stream, 'operation': 'refresh_' + event, 'seconds': time.perf_counter() - timer})
            acquired.update(fresh_global)
            receipt['events'][event] = {'announced_global_rows': announced_global, 'announced_state_rows': rows.tolist(),
                                        'announced_image_ids': [x['image_id'] for x in requested], 'fresh_global_rows': fresh_global,
                                        'already_known_global_rows': already_global, 'new_unique_target_rows': len(fresh_global),
                                        'refresh_before_label_release': True, 'authoritative': True,
                                        'requested_augmented_payload_sha256': hashlib.sha256(values.tobytes(order='C')).hexdigest()}
            capture('after_refresh_' + event)
            changes = []
            for x, i in zip(requested, rows):
                changes.append(LabelChange(int(i), int(x['expected_label_version']), int(x['old_label']), int(x['new_label'])))
                request_export.append({'stream': stream, 'event': event, 'state_row': int(i), 'global_row': int(x['row']),
                                       'image_id': x['image_id'], 'stage': x['stage'], 'old_label': int(x['old_label']),
                                       'new_label': int(x['new_label']), 'expected_label_version': int(x['expected_label_version']),
                                       'new_label_version': int(x['new_label_version'])})
            timer = time.perf_counter()
            state.relabel(changes, feature_version='v2')
            costs.append({'stream': stream, 'operation': 'label_release_' + event, 'seconds': time.perf_counter() - timer})
            if any(state.versions[int(i)] != x['new_label_version'] for x, i in zip(requested, rows)):
                raise RuntimeError('Frozen version transition differs')
            capture('after_' + event)
        receipt['candidate_unique_final_rows'] = sorted(acquired)
        receipt['unique_target_acquisition_counts'] = {'initial_fit_pilot': 1024, 'R1_new_history': receipt['events']['R1']['new_unique_target_rows'],
                                                       'E_append': len(arrivals), 'R2_new_history': receipt['events']['R2']['new_unique_target_rows'],
                                                       'total_unique': len(acquired)}
        receipt['target_bank_roles'] = bank.calls
        receipt['fold5_in_candidate_evidence'] = False
        receipt['reference_weights_are_evaluator_only'] = True
        receipts[stream] = receipt
        write(out / 'STATE_TIMELINE.json', {'stream': stream, 'stages': discrete, 'history_global_rows': history,
                                           'E_global_rows': arrivals, 'fit_global_rows': fit_global, 'pilot_global_rows': pilot_global})
        for path in out.iterdir():
            rel = path.relative_to(ROOT).as_posix()
            if rel not in files:
                files[rel] = array_identity(path) if path.suffix == '.npz' else identity(path)
        print(json.dumps({'event': 'ALL_STREAM_STATES_BUILT_WITHOUT_SCORING', 'stream': stream, 'utc': utc()}), flush=True)
    write(ROOT / 'ACCESS_RECEIPTS.json', {'status': 'COMMON_EVIDENCE_IDENTICAL_RECEIPTS_FROZEN', 'streams': receipts,
                                         'fold0_or_fold1_in_confirmation': False, 'fold5_candidate_evidence': False})
    csvwrite(ROOT / 'REQUEST_MANIFEST_ALL.csv', request_export)
    write(ROOT / 'BUILD_COST_ACCOUNTING.json', {'started_utc': started, 'completed_utc': utc(), 'operations': costs,
                                               'seconds': time.perf_counter() - clock, 'frontend_locks': frontend_locks})
    BOUNDARY.assert_clean()
    write(ROOT / 'audit/BUILD_ACCESS_EVIDENCE.json', {'status': 'NO_FORBIDDEN_BUILD_ACCESS_PASS', 'forbidden': BOUNDARY.forbidden,
                                                     'feature_reads': BOUNDARY.feature_reads, 'candidate_performance_computed': False,
                                                     'fold5_candidate_evidence': False, 'reference_compartment': 'stage full-target W only; no evaluator scores',
                                                     'torch_or_torchvision_imported': False,
                                                     'housekeeping_metadata_denials': BOUNDARY.housekeeping_metadata_denials,
                                                     'actual_network_connections': 0, 'actual_external_processes': 0, 'actual_NUL_writes': 0})
    for rel in ('ACCESS_RECEIPTS.json', 'REQUEST_MANIFEST_ALL.csv', 'BUILD_COST_ACCOUNTING.json', 'audit/BUILD_ACCESS_EVIDENCE.json'):
        files[rel] = identity(ROOT / rel)
    lock = {'status': 'ALL_FRONTENDS_CANDIDATES_AND_STAGES_FROZEN_BEFORE_PERFORMANCE', 'started_utc': started, 'frozen_utc': utc(),
            'encoder_feature_lock_identity': identity(ROOT / 'ENCODER_FEATURE_LOCK.json'), 'frontend_locks': frontend_locks,
            'files': files, 'method_contract_identity': identity(ROOT / 'ARTIFACT_CONTRACT.json'),
            'current_core_sha256': CORE_SHA, 'procrustes_sha256': PROCRUSTES_SHA, 'all_18_states_complete': True,
            'fold5_performance_computed': False, 'response_metrics_computed': False}
    write(ROOT / 'BUILD_LOCK.json', lock)
    print(json.dumps({'event': lock['status'], 'utc': utc()}), flush=True)


def rms(x):
    return float(np.sqrt(np.mean(np.square(x))))


def relative(e, ref):
    return float(np.linalg.norm(e) / max(float(np.linalg.norm(ref)), 1e-15))


def retrieval(v, g, yq, yg):
    # Exact accepted bridge function; metric-only normalization never alters proxies.
    v = v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-15)
    g = g / np.maximum(np.linalg.norm(g, axis=1, keepdims=True), 1e-15)
    order = np.argsort(-(v @ g.T), axis=1, kind='stable')
    relevant = yg[order] == yq[:, None]
    den = relevant.sum(1)
    ap = ((np.cumsum(relevant, axis=1) / (1 + np.arange(len(g)))) * relevant).sum(1) / np.maximum(den, 1)
    return dict(top1=float(relevant[:, 0].mean()), map=float(ap[den > 0].mean()), queries_with_relevant=int((den > 0).sum()))


def verify_build():
    lock = read(ROOT / 'BUILD_LOCK.json')
    if lock['status'] != 'ALL_FRONTENDS_CANDIDATES_AND_STAGES_FROZEN_BEFORE_PERFORMANCE' or not lock['all_18_states_complete']:
        raise RuntimeError('All-state build gate not PASS')
    if identity(ROOT / 'ENCODER_FEATURE_LOCK.json') != lock['encoder_feature_lock_identity']:
        raise RuntimeError('Eight cache lock changed')
    for rel, item in lock['files'].items():
        p = ROOT / rel
        if p.stat().st_size != item['bytes'] or sha(p) != item['sha256']:
            raise RuntimeError('Build-locked asset changed: ' + rel)
    for stream, item in lock['frontend_locks'].items():
        if identity(ROOT / f'FRONTEND_LOCKS/{stream}.json') != item:
            raise RuntimeError('Frontend lock changed')
    repair = read(ROOT / 'audit/BUILD_REPAIR_SEMANTIC_IDENTITY.json')
    if (repair['status'] != 'BUILD_REPAIR_NUMERICAL_SEMANTICS_IDENTICAL'
            or repair['compared_npz_count'] != 24 or repair['expected_npz_count'] != 24
            or repair['failure'] is not None or len(repair['comparisons']) != 24):
        raise RuntimeError('All-24 pre-performance repair semantic gate not PASS')
    if repair['repaired_build_lock_identity'] != identity(ROOT / 'BUILD_LOCK.json'):
        raise RuntimeError('Repair semantic gate is not bound to current frozen build')
    expected_npz = {rel for rel in lock['files'] if rel.endswith('.npz')}
    if set(repair['comparisons']) != expected_npz:
        raise RuntimeError('Repair semantic gate does not cover exact current 24 NPZ')
    for rel, item in repair['comparisons'].items():
        if (not item['all_array_equal'] or not item['semantic_identity_equal']
                or not all(item['array_equal'].values())
                or item['current_identity'] != identity(ROOT / rel)):
            raise RuntimeError('Repair semantic check or current asset identity failure; STOP')
    return lock


def evaluate():
    verify_sources()
    feature_lock, data = verify_all_features()
    build_lock = verify_build()
    started, clock = utc(), time.perf_counter()
    if started <= build_lock['frozen_utc']:
        raise RuntimeError('Evaluation precedes all-state freeze')
    role('evaluation')
    evaluator = load_numeric(ROOT / 'features/swin_t/fold5.npz')
    v, truth = aug(evaluator['features']), evaluator['labels']
    stage_metrics, event_metrics, frontend_metrics, evaluation_cost, score_locks = [], [], [], [], {}
    receipts = read(ROOT / 'ACCESS_RECEIPTS.json')
    for stream, fold in STREAMS.items():
        timer = time.perf_counter()
        out = ROOT / f'scores/{stream}'
        out.mkdir(exist_ok=False)
        scores, predictions, states = {}, {}, {}
        for stage in STAGES:
            state = load_numeric(ROOT / f'states/{stream}/{stage}.npz')
            states[stage] = state
            reference_weight = state['w_full_target_ridge']
            reference_score = v @ reference_weight
            for method, wkey in (('mean', 'w_mean'), ('paired025', 'w_paired025'), ('sample_current', 'w_sample'), ('full_target_ridge', 'w_full_target_ridge')):
                score = reference_score if method == 'full_target_ridge' else v @ state[wkey]
                pred = score.argmax(1)
                key = stage + '__' + method
                scores[key], predictions[key] = score, pred
                known = int(state['known'].sum())
                stage_metrics.append({'stream': stream, 'stage': stage, 'method': method, 'correct': int((pred == truth).sum()),
                                      'count': len(truth), 'accuracy': float(np.mean(pred == truth)), 'score_rms': rms(score - reference_score),
                                      'weight_rel': relative(state[wkey] - reference_weight, reference_weight),
                                      'known_rows': len(state['labels']) if method == 'full_target_ridge' else known,
                                      'unique_candidate_target_rows': known if method != 'full_target_ridge' else 0,
                                      'reference_target_rows': len(state['labels']) if method == 'full_target_ridge' else 0,
                                      'reference_only': method == 'full_target_ridge'})
        request_lock = read(PROTO / f'REQUEST_LOCKS/{stream}.json')
        for event in ('R1', 'R2'):
            ref = scores['after_' + event + '__full_target_ridge'] - scores['after_refresh_' + event + '__full_target_ridge']
            ref_action = scores['after_' + event + '__full_target_ridge'] - scores['before_' + event + '__full_target_ridge']
            if not np.array_equal(ref, ref_action):
                raise RuntimeError('Reference refresh changed labels or target reference')
            for method in METHODS:
                delta = scores['after_' + event + '__' + method] - scores['after_refresh_' + event + '__' + method]
                total = scores['after_' + event + '__' + method] - scores['before_' + event + '__' + method]
                event_metrics.append({'stream': stream, 'event': event, 'method': method,
                                      'response_rel': relative(delta - ref, ref), 'response_error_rms': rms(delta - ref),
                                      'reference_response_rms': rms(ref), 'total_action_rel': relative(total - ref, ref),
                                      'total_error_rms': rms(total - ref), 'reference_action_rms': rms(ref_action),
                                      'request_rows': request_lock['requests'][event]['count']})
        # Descriptive frontend qualification opens only after all candidate states are immutable.
        initial = states['before_R1']
        hold = np.ones(len(initial['labels']), dtype=bool)
        front = read(ROOT / f'FRONTEND_LOCKS/{stream}.json')
        hold[np.asarray(front['fit_state_rows'], dtype=np.int64)] = False
        hold[np.asarray(front['pilot_state_rows'], dtype=np.int64)] = False
        raw = load_numeric(ROOT / f'features/swin_t/fold{fold}.npz')
        raw_index = {int(r): i for i, r in enumerate(raw['rows'])}
        target = raw['features'][[raw_index[int(r)] for r in initial['global_rows'][hold]]].astype(np.float64)
        target_labels = raw['labels'][[raw_index[int(r)] for r in initial['global_rows'][hold]]]
        z = initial['proxy'][hold, :-1]
        zc, tc = z - z.mean(0), target - target.mean(0)
        rr = retrieval(evaluator['features'].astype(np.float64), z, truth, target_labels)
        zd, td = len(np.unique(z, axis=0)), len(np.unique(target, axis=0))
        before_accuracy = next(x['accuracy'] for x in stage_metrics if x['stream'] == stream and x['stage'] == 'before_R1' and x['method'] == 'mean')
        frontend_metrics.append({'stream': stream, 'hold_count': len(z), 'feature_rms': rms(z - target),
                                 'cosine_mean': float(np.mean(np.sum(z * target, axis=1) / np.maximum(np.linalg.norm(z, axis=1) * np.linalg.norm(target, axis=1), 1e-15))),
                                 'same_class_retrieval_top1': rr['top1'], 'same_class_map': rr['map'],
                                 'retrieval_query_count': len(truth), 'queries_with_relevant': rr['queries_with_relevant'],
                                 'centered_output_rms': rms(zc), 'centered_target_rms': rms(tc),
                                 'centered_output_target_ratio': rms(zc) / rms(tc),
                                 'covariance_trace_ratio': float(np.sum(zc * zc) / np.sum(tc * tc)),
                                 'exact_proxy_distinct': zd, 'exact_proxy_duplicates': len(z) - zd,
                                 'exact_target_distinct': td, 'exact_target_duplicates': len(target) - td,
                                 'before_R1_mean_accuracy': before_accuracy, 'outcome_dependent_changes': False})
        sid, pid = save(out / 'SCORES.npz', scores), save(out / 'PREDICTIONS.npz', predictions)
        score_locks[stream] = {'scores': sid, 'predictions': pid}
        evaluation_cost.append({'stream': stream, 'seconds': time.perf_counter() - timer,
                                'retained_scores_bytes': sum(x.nbytes for x in scores.values()),
                                'prediction_array_bytes': sum(x.nbytes for x in predictions.values())})
    csvwrite(ROOT / 'STAGE_METRICS.csv', stage_metrics)
    csvwrite(ROOT / 'REQUESTS_ALL.csv', event_metrics)
    csvwrite(ROOT / 'FRONTEND_QUALIFICATION.csv', frontend_metrics)
    write(ROOT / 'FRONTEND_QUALIFICATION.json', {'status': 'DESCRIPTIVE_ONLY_FROZEN_FRONTEND_QUALIFICATION', 'streams': frontend_metrics,
                                                'retrieval_definition': read(ROOT / 'ARTIFACT_CONTRACT.json')['frontend_qualification']['retrieval'],
                                                'does_not_change_frontend_or_remove_stream': True})
    averages, directions = {}, []
    for event in ('R1', 'R2'):
        m = [x for x in event_metrics if x['event'] == event and x['method'] == 'mean']
        p = [x for x in event_metrics if x['event'] == event and x['method'] == 'paired025']
        mr, pr = float(np.mean([x['response_rel'] for x in m])), float(np.mean([x['response_rel'] for x in p]))
        mt, pt = float(np.mean([x['total_action_rel'] for x in m])), float(np.mean([x['total_action_rel'] for x in p]))
        averages[event] = {'mean_response_rel': mr, 'paired025_response_rel': pr, 'paired025_response_lower': pr < mr,
                            'mean_total_action_rel': mt, 'paired025_total_action_rel': pt, 'paired025_total_action_lower': pt < mt}
        for stream in STREAMS:
            a, b = next(x for x in m if x['stream'] == stream), next(x for x in p if x['stream'] == stream)
            directions.append({'stream': stream, 'event': event, 'mean_response_rel': a['response_rel'], 'paired025_response_rel': b['response_rel'],
                               'paired025_response_lower': b['response_rel'] < a['response_rel'],
                               'signed_relative_reduction': 1 - b['response_rel'] / a['response_rel'],
                               'mean_total_action_rel': a['total_action_rel'], 'paired025_total_action_rel': b['total_action_rel'],
                               'paired025_total_action_lower': b['total_action_rel'] < a['total_action_rel']})
    passed = sum(x['paired025_response_lower'] for x in averages.values())
    status = ('FOOD101_EXTERNAL_PRIMARY_NOT_CONFIRMED', 'FOOD101_EXTERNAL_PRIMARY_PARTIAL', 'FOOD101_EXTERNAL_PRIMARY_CONFIRMED_BOTH')[passed]
    final = {method: float(np.mean([x['accuracy'] for x in stage_metrics if x['stage'] == 'after_R2' and x['method'] == method])) for method in METHODS}
    primary = {'status': status, 'event_averages': averages, 'all_six_stream_event_directions': directions,
               'final_stage_average_accuracy': final, 'equal_stream_weight': True, 'minimum_effect_size_threshold': None,
               'p_values_computed': False, 'automatic_next_experiment': False, 'evaluation_started_utc': started, 'evaluation_completed_utc': utc(),
               'build_lock_identity': identity(ROOT / 'BUILD_LOCK.json')}
    write(ROOT / 'PRIMARY_RESULT.json', primary)
    write(ROOT / 'SCORE_LOCK.json', {'status': 'ALL_CONFIRMATION_SCORES_AND_PREDICTIONS_FROZEN', 'streams': score_locks,
                                    'primary_result_identity': identity(ROOT / 'PRIMARY_RESULT.json'), 'frozen_utc': utc()})
    write(ROOT / 'COST_ACCOUNTING.json', {'physical_feature_cache_creation': feature_lock,
                                         'frontend_fit_and_history_map': {s: read(ROOT / f'FRONTEND_LOCKS/{s}.json') for s in STREAMS},
                                         'current_evidence_and_reference_build': read(ROOT / 'BUILD_COST_ACCOUNTING.json'),
                                         'candidate_logical_target_access': {s: receipts['streams'][s]['unique_target_acquisition_counts'] for s in STREAMS},
                                         'evaluation_and_full_reference_cache': evaluation_cost, 'evaluation_seconds': time.perf_counter() - clock,
                                         'blas_threads': 8, 'threadpools': threadpool_info(), 'speed_ranking_claim': False})
    BOUNDARY.assert_clean()
    write(ROOT / 'audit/EVALUATION_ACCESS_EVIDENCE.json', {'status': 'NO_FORBIDDEN_EVALUATION_ACCESS_PASS', 'forbidden': BOUNDARY.forbidden,
                                                          'feature_reads': BOUNDARY.feature_reads, 'model_or_image_access': False,
                                                          'no_frontend_or_candidate_refit': True})
    write(ROOT / 'NO_FORBIDDEN_ACCESS_AUDIT.json', {'status': 'NO_FORBIDDEN_CONFIRMATION_ACCESS_PASS',
                                                  'build_access_evidence': identity(ROOT / 'audit/BUILD_ACCESS_EVIDENCE.json'),
                                                  'evaluation_access_evidence': identity(ROOT / 'audit/EVALUATION_ACCESS_EVIDENCE.json'),
                                                  'allowed_confirmation_folds': [2, 3, 4], 'common_evaluator_fold': 5,
                                                  'fold0_or_fold1_confirmation_access': False, 'fold5_candidate_evidence': False,
                                                  'encoder_reselection': False, 'resplit_or_reseed': False, 'alpha_changed': False,
                                                  'corruption_changed': False, 'requests_changed': False, 'FastFill_training': False,
                                                  'variant_search_or_rescue': False, 'automatic_next_experiment': False})
    print(json.dumps(primary), flush=True)


def main():
    global BOUNDARY
    parser = argparse.ArgumentParser()
    parser.add_argument('phase', choices=('build', 'evaluate'))
    args = parser.parse_args()
    BOUNDARY = AccessBoundary(args.phase)
    sys.addaudithook(BOUNDARY.audit)
    try:
        with threadpool_limits(limits=8, user_api='blas'):
            (build if args.phase == 'build' else evaluate)()
    except BaseException as error:
        path = ROOT / ('BUILD_FAILURE.json' if args.phase == 'build' else 'EVALUATION_FAILURE.json')
        if not path.exists():
            write(path, {'status': 'STOP', 'phase': args.phase, 'utc': utc(), 'error': repr(error), 'forbidden_access': BOUNDARY.forbidden,
                         'after_candidate_performance_bug_invalidates_run': args.phase == 'evaluate'})
        raise


if __name__ == '__main__':
    main()
