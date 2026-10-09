"""Authenticate the accepted protocol; derive only a candidate-visible view."""
import zipfile
import hashlib
import numpy as np
from common_io import ROOT, PROTOCOL, TASK, read, write, asset, sha, selected_json_fields

ZIP = PROTOCOL/'deliverables/011_fastfill_final_unseen_protocol_freeze_01_20261006.zip'
ZIP_SHA = '744ebb41dccaa2a99e16c878b01c2c1dd72b4a1c2850f906a294c50f7d77355c'

def run():
    assert ZIP.stat().st_size == 1490420 and sha(ZIP) == ZIP_SHA
    with zipfile.ZipFile(ZIP) as z:
        assert len(z.namelist()) == 46 and z.testzip() is None
        manifest = __import__('json').loads(z.read('MANIFEST.json'))
        for n, rec in manifest['files'].items():
            b = z.read(n)
            assert len(b) == rec['bytes'] and hashlib.sha256(b).hexdigest() == rec['sha256']
            assert (PROTOCOL/n).read_bytes() == b, n
    names = ['RESERVE_IDENTITY.json','FINAL_STREAM_SPLIT_LOCK.json','FIT_PILOT_LOCK.json',
             'REQUESTS_LOCK.json','METHOD_LOCK.json','SOURCE_PROTOCOL_IDENTITY.json','CLAIM_FREEZE.md']
    split = read(PROTOCOL/'FINAL_STREAM_SPLIT_LOCK.json')
    evidence = read(PROTOCOL/'FIT_PILOT_LOCK.json'); method = read(PROTOCOL/'METHOD_LOCK.json')
    reserve = read(PROTOCOL/'RESERVE_IDENTITY.json')
    assert reserve['status'] == 'PASS' and reserve['reserve']['count'] == 6000
    assert not reserve['reserve_contamination_evidence'] and not reserve['unverifiable_cache_namespaces']
    assert read(PROTOCOL/'audit/INDEPENDENT_PROTOCOL_AUDIT.json')['status'] == 'PASS'
    assert read(PROTOCOL/'DELIVERY_STATUS.json')['status'] == 'PROTOCOL_FREEZE_PASS'
    assert split['seed'] == evidence['seed'] == 907001
    ids = np.asarray(split['selected_ids'],dtype=np.int64)
    assert len(ids) == len(set(ids.tolist())) == 6000
    assert set(ids.tolist()) == set(reserve['reserve']['ids'])
    for name, bounds in [('history',(0,4000)),('arrival_E',(4000,5000)),('evaluation',(5000,6000))]:
        g = split['groups'][name]; a,b = bounds
        assert g['rows'] == list(range(a,b)) and g['global_ids'] == ids[a:b].tolist()
    fit = evidence['fit']['rows']; pilot = evidence['pilot']['rows']
    assert len(fit) == len(pilot) == 512 and len(set(fit+pilot)) == 1024
    assert evidence['fit']['global_ids'] == ids[fit].tolist() and evidence['pilot']['global_ids'] == ids[pilot].tolist()
    assert method['backend'] == dict(methods=['mean','paired025'],alpha=.25,ridge=1,shrink=.1,
        current_core_sha256='0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7',current_core_executed=False)
    assert method['frontend']['checkpoint_sha256'] == '0ac692e1e9dd0994e89a3f40fbc165a34cd097503ded8ac56cea0249712538f8'
    # Selective reader leaves evaluation_truth opaque until evaluation phase.
    stages = selected_json_fields(PROTOCOL/'REQUESTS_LOCK.json', ['stage_labels'])['stage_labels']
    public = dict(task=TASK,protocol_seed=907001,stream_count=1,split=split,fit_pilot=evidence,
        method=method,initial_history_labels=stages['before_R1']['labels'],
        initial_history_versions=stages['before_R1']['versions'],
        initial_E_labels=stages['before_R2']['labels'][4000:],request_counts=dict(R1=819,R2=1228),
        evaluator_truth_exposed=False,request_new_labels_exposed=False)
    assert len(public['initial_history_labels']) == 4000 and len(public['initial_E_labels']) == 1000
    assert not any(public['initial_history_versions'])
    write(ROOT/'PUBLIC_PROTOCOL.json', public)
    write(ROOT/'PROTOCOL_GATE.json',dict(task=TASK,status='PASS',accepted_status='FINAL_UNSEEN_PROTOCOL_ACCEPTED_FOR_ONE_SHOT_EXECUTION',
        protocol_zip=asset(ZIP),protocol_assets={n:asset(PROTOCOL/n) for n in names},
        public_protocol=asset(ROOT/'PUBLIC_PROTOCOL.json'),stream_id=907001,stream_count=1,
        split_rng_called=False,fit_pilot_rng_called=False,request_rng_called=False,
        evaluation_truth_deserialized=False,old_protocol_modified=False))
    print('{"status":"PASS","protocol_seed":907001,"RNG_regeneration":false,"eval_truth_exposed":false}')

if __name__ == '__main__': run()
