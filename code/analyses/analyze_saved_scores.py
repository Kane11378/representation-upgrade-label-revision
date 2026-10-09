"""Read-only decomposition of the frozen FastFill bridge scores.

No model, feature cache, labels, fitting, alpha selection, or additional inference.
Usage: python -B analyze_saved_scores.py --root COMPLETED_BRIDGE --out NEW_DIR
       python -B analyze_saved_scores.py --self-test
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

SEEDS = (905101, 905102, 905103)
STAGES = ('before_R1','after_refresh_R1','after_R1','before_R2','after_refresh_R2','after_R2')
M, P, S, F = 'fastfill_norm__mean','fastfill_norm__paired025','sample_current','full_target_ridge'
SCORE_HASHES = {
    905101:'62a257601262d26c480c70b548ba2828a1972b15f8ac823754469ec28bcc18df',
    905102:'47af38ff432e40e4f1dc008bc5d3e252f45bb238332b6bff90442626a8478307',
    905103:'edacd54dddd01df20a7606ff20be13d8c23dc05ea97f931f5c9bb530aa61ed3a',
}
# The table digests below are inserted from the accepted delivery, not from new results.
TABLE_HASHES = {'REQUESTS_ALL.csv': '3b88a4019fbd710c18fb233ce868ae02aeb0dd7746a424a7a6d65307583aec6e', 'EVENTS_ALL.csv': '651b440e1627607a12b03824b7ed020055f40af3e2e7f69e584536eeb1920a7a', 'EXTERNAL_LARGE_ARTIFACTS.json': 'dcfe48cd7f65390efb3e9bdd02810a2f66ad10ad76758de4854727c2e99cf1a9'}
ATOL = RTOL = 1e-10

def sha(p: Path) -> str:
    h=hashlib.sha256()
    with p.open('rb') as stream:
        for block in iter(lambda:stream.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def ensure(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)

def close(a, b, message: str) -> None:
    ensure(bool(np.allclose(a,b,atol=ATOL,rtol=RTOL)),message)

def energy(a: np.ndarray) -> float: return float(np.mean(a*a))

def decomposition(before, refreshed, after, full_before, full_refreshed, full_after):
    close(full_before,full_refreshed,'Oracle changed during feature-only refresh')
    reference=full_after-full_refreshed
    refresh=refreshed-before
    label_error=after-refreshed-reference
    total=after-before-(full_after-full_before)
    e0=before-full_before; e1=after-full_after
    close(total,refresh+label_error,'total != refresh + label error')
    close(e1,e0+total,'endpoint identity failed')
    result=dict(reference_energy=energy(reference),refresh_energy=energy(refresh),
        label_error_energy=energy(label_error),refresh_label_interaction=float(2*np.mean(refresh*label_error)),
        total_error_energy=energy(total),initial_score_error_energy=energy(e0),
        initial_total_interaction=float(2*np.mean(e0*total)),final_score_error_energy=energy(e1))
    close(result['total_error_energy'],result['refresh_energy']+result['label_error_energy']+result['refresh_label_interaction'],'maintenance energy identity failed')
    close(result['final_score_error_energy'],result['initial_score_error_energy']+result['total_error_energy']+result['initial_total_interaction'],'endpoint energy identity failed')
    ensure(result['reference_energy']>0,'zero reference event')
    return result

def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def read_table(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig',newline='') as stream:return list(csv.DictReader(stream))

def self_test() -> None:
    rng=np.random.default_rng(20261006)
    shape=(7,5);f0=rng.normal(size=shape);f1=f0+rng.normal(size=shape)
    a=[rng.normal(size=shape) for _ in range(3)]
    d=decomposition(*a,f0,f0.copy(),f1)
    ensure(d['total_error_energy']>=0,'negative energy')
    # A better final estimate can coexist with a worse change estimate.
    zeros=np.zeros((2,2));ones=np.ones((2,2))
    q=decomposition(2*ones,ones,ones,zeros,zeros,ones)
    close(q['final_score_error_energy'],0.,'exact endpoint test')
    ensure(q['total_error_energy']>0,'nonzero change error test')
    try: decomposition(*a,f0,f0+1,f1)
    except ValueError: pass
    else: raise AssertionError('modified oracle refresh was not rejected')
    print('SELF_TEST_PASS: algebra, endpoint/change distinction, invalid oracle rejection')

def run(root: Path, out: Path) -> None:
    ensure(root.is_dir(),'Completed bridge root is missing')
    ensure(not out.exists(),'Output directory must be new; do not overwrite')
    ensure(not out.resolve().is_relative_to(root.resolve()),'Output must be outside completed bridge directory')
    out.mkdir(parents=True)
    status={'task':'FASTFILL_BRIDGE_MAINTENANCE_DECOMPOSITION_01','scope':'Post-hoc read-only analysis of fixed saved scores, not a new scientific confirmation.'}
    try:
        watched={}
        for name,digest in TABLE_HASHES.items():
            path=root/name;ensure(sha(path)==digest,'Table identity mismatch: '+name);watched[name]=(path.stat().st_mtime_ns,digest)
        requests={(int(x['seed']),x['event'],x['method']):x for x in read_table(root/'REQUESTS_ALL.csv')}
        events={(int(x['seed']),x['stage'],x['method']):x for x in read_table(root/'EVENTS_ALL.csv')}
        rows=[];deltas=[]
        for seed in SEEDS:
            name=f'evaluations/{seed}/SCORES_{seed}.npz';path=root/name
            ensure(sha(path)==SCORE_HASHES[seed],'Saved score hash mismatch: '+name)
            watched[name]=(path.stat().st_mtime_ns,SCORE_HASHES[seed])
            with np.load(path,allow_pickle=False) as pack:
                expected={stage+'__'+method for stage in STAGES for method in (M,P,S,F)}
                ensure(set(pack.files)==expected,'Score array names differ')
                a={k:pack[k] for k in pack.files}
            for key,v in a.items():ensure(v.shape==(1000,100) and v.dtype==np.float64 and np.isfinite(v).all(),'Invalid score array: '+key)
            for stage in STAGES:
                close(a[stage+'__'+P],.75*a[stage+'__'+M]+.25*a[stage+'__'+S],'Fixed .25 mixture identity failed')
                for method in (M,P,S,F):
                    value=np.sqrt(energy(a[stage+'__'+method]-a[stage+'__'+F]))
                    close(value,float(events[(seed,stage,method)]['score_rms']),'Stored stage score RMS mismatch')
            for event in ('R1','R2'):
                before,refresh,after='before_'+event,'after_refresh_'+event,'after_'+event
                bymethod={}
                for method in (M,P,S):
                    d=decomposition(*(a[t+'__'+method] for t in (before,refresh,after)),*(a[t+'__'+F] for t in (before,refresh,after)))
                    old=requests[(seed,event,method)]
                    for field,key in [('response_error_rms','label_error_energy'),('total_error_rms','total_error_energy'),('reference_response_rms','reference_energy')]:close(np.sqrt(d[key]),float(old[field]),'Stored request RMS mismatch')
                    close(np.sqrt(d['label_error_energy']/d['reference_energy']),float(old['response_rel']),'Stored relative response mismatch')
                    close(np.sqrt(d['total_error_energy']/d['reference_energy']),float(old['total_action_rel']),'Stored relative total mismatch')
                    rows.append(dict(seed=seed,event=event,method=method,**d));bymethod[method]=d
                left,right=bymethod[M],bymethod[P]
                dr={k:right[k]-left[k] for k in right if k!='reference_energy'}
                close(dr['total_error_energy'],dr['refresh_energy']+dr['label_error_energy']+dr['refresh_label_interaction'],'Difference identity failed')
                deltas.append(dict(seed=seed,event=event,**{'delta_'+k:v for k,v in dr.items()},**{'normalized_delta_'+k:v/left['reference_energy'] for k,v in dr.items()}))
        for name,(mtime,digest) in watched.items():ensure((root/name).stat().st_mtime_ns==mtime and sha(root/name)==digest,'Source file changed: '+name)
        summary=[]
        for event in ('R1','R2'):
            selected=[r for r in deltas if r['event']==event]
            summary.append(dict(event=event,**{k:float(np.mean([r[k] for r in selected])) for k in selected[0] if k not in ('seed','event')}))
        write_csv(out/'REQUEST_ERROR_DECOMPOSITION.csv',rows)
        write_csv(out/'PAIRED_MINUS_MEAN.csv',deltas)
        write_csv(out/'DECOMPOSITION_SUMMARY.csv',summary)
        status.update(status='PASS',identity_files=len(watched),seeds=list(SEEDS),decomposed_rows=len(rows),fixed_alpha=.25,changed_source_files=0,
            not_performed=['model loading','feature cache access','label access','training','inference','new estimators','alpha search','sigma selection','reserve access'],summary=summary)
        (out/'REPORT.md').write_text('# Maintenance-error decomposition of saved scores\n\n'
            'The read-only recomputation is complete. No new training, inference, parameter selection, or independent confirmation was performed.\n\n'
            'REQUEST_ERROR_DECOMPOSITION.csv decomposes total maintenance-error energy into feature-refresh energy, label-response-error energy, and their interaction.\n'
            'In PAIRED_MINUS_MEAN.csv, positive differences indicate a larger contribution for paired than for mean; negative differences indicate a smaller contribution. The component differences sum to the total maintenance-error difference.\n'
            'The score-error identity is also checked: final error = initial error + total maintenance error. A closer endpoint and a more accurate maintenance action are therefore distinct conditions.\n'
            'These results diagnose existing development streams after the fact. They do not establish independent replication or statistical significance. Full numerical results are in the CSV files and STATUS.json.\n',encoding='utf-8')
    except Exception as exc:
        status.update(status='FAIL',error=type(exc).__name__+': '+str(exc),STOP=True)
        raise
    finally:
        (out/'STATUS.json').write_text(json.dumps(status,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print('PASS; read-only decomposition complete. STOP.')

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path);parser.add_argument('--out',type=Path);parser.add_argument('--self-test',action='store_true')
    args=parser.parse_args()
    if args.self_test:self_test()
    else:
        if args.root is None or args.out is None:parser.error('--root and --out are required')
        run(args.root,args.out)
