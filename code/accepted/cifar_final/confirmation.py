"""One frozen stream. Candidate build precedes all evaluator/performance access."""
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from common_io import (ROOT, PROTOCOL, TASK, STAGES, METHODS, ALL_METHODS, read,
                       write, sha, asset, verify, semantic, source_guard, selected_json_fields)

CORE_SHA = '0b28ae97c4fcd65eb901938e06a8642a899e5c32208b239da7b4ac64101d6bd7'

# Byte-identical original bridge numerical helpers.
def aug(x):return np.column_stack((x,np.ones(len(x))))
def rms(x):return float(np.sqrt(np.mean(np.square(x))))
def relative(e,ref):return float(np.linalg.norm(e)/max(float(np.linalg.norm(ref)),1e-15))
def retrieval(v,g,yq,yg):
 v=v/np.maximum(np.linalg.norm(v,axis=1,keepdims=True),1e-15);g=g/np.maximum(np.linalg.norm(g,axis=1,keepdims=True),1e-15)
 order=np.argsort(-(v@g.T),axis=1,kind='stable');relevant=yg[order]==yq[:,None];den=relevant.sum(1)
 ap=((np.cumsum(relevant,axis=1)/(1+np.arange(len(g))))*relevant).sum(1)/np.maximum(den,1)
 return dict(top1=float(relevant[:,0].mean()),map=float(ap[den>0].mean()),queries_with_relevant=int((den>0).sum()))

def now(): return datetime.now(timezone.utc).isoformat()

def writecsv(path, rows):
    assert rows
    with Path(path).open('x', encoding='utf-8', newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def imports_core():
    assert sha(ROOT/'code/current_core.py') == CORE_SHA
    from current_core import CurrentEvidenceState, LabelChange, solve, cross
    return CurrentEvidenceState, LabelChange, solve, cross

def gates():
    source_guard()
    for n in ['FEATURE_CACHE_LOCK.json','FASTFILL_PROXY_LOCK.json']:
        d=read(ROOT/n)
        assert d['status'].startswith('FROZEN') or d['status'] in ['PASS','LOCKED']
        for rec in d['assets'].values(): verify(rec)
    assert read(ROOT/'DINO_IDENTITY_GATE.json')['status']=='PASS'
    p=read(ROOT/'PUBLIC_PROTOCOL.json')
    assert p['protocol_seed']==907001 and p['stream_count']==1
    assert p['method']['backend']['alpha']==.25
    assert p['method']['backend']['ridge']==1 and p['method']['backend']['shrink']==.1
    return p

class CandidateTargetProvider:
    """B cache is infrastructure; candidate receives only authorized row slices."""
    def __init__(self, cache, ids):
        self.cache=cache;self.ids=ids;self.allowed=set();self.access_log=[]
    def authorize(self, rows, event):
        rows=np.asarray(rows,dtype=np.int64)
        assert np.all((rows>=0)&(rows<5000)) and len(set(rows.tolist()))==len(rows)
        fresh=[r for r in rows.tolist() if r not in self.allowed]
        self.allowed.update(rows.tolist())
        self.access_log.append(dict(event=event,requested_rows=rows.tolist(),new_rows=fresh,
                                    global_ids=self.ids[rows].tolist()))
        return aug(self.cache[rows].astype(np.float64))

def build():
    p=gates();assert not (ROOT/'BUILD_LOCK.json').exists()
    assert not (ROOT/'evaluation/PERFORMANCE_ACCESS_STARTED.json').exists()
    State, Change, _, _=imports_core();start=time.perf_counter()
    feature=read(ROOT/'FEATURE_CACHE_LOCK.json');proxy_lock=read(ROOT/'FASTFILL_PROXY_LOCK.json')
    ids=np.load(verify(feature['assets']['global_ids']),allow_pickle=False)
    assert ids.tolist()==p['split']['selected_ids']
    cache=np.load(verify(feature['assets']['B']),mmap_mode='r',allow_pickle=False)
    assert cache.shape==(6000,768) and cache.dtype==np.dtype('float32')
    raw=np.load(verify(proxy_lock['assets']['proxy']),allow_pickle=False)
    assert raw.shape==(4000,768) and raw.dtype==np.dtype('float32')
    proxy=aug(raw.astype(np.float64))
    fit=np.asarray(p['fit_pilot']['fit']['rows'],dtype=np.int64)
    pilot=np.asarray(p['fit_pilot']['pilot']['rows'],dtype=np.int64)
    initial=np.r_[fit,pilot]
    provider=CandidateTargetProvider(cache,ids)
    observed=np.zeros_like(proxy);observed[initial]=provider.authorize(initial,'initial_fit_pilot')
    known=np.zeros(4000,bool);known[initial]=True
    certainty=np.zeros(4000,bool);certainty[fit]=True
    state=State(proxy,np.asarray(p['initial_history_labels'],dtype=np.int64),known,observed,
                certainty,pilot,100,1.,.1,'v2')
    heads={};arrays={};stage_receipts={};state_bytes={};events={};release_log=[]
    locked_stages=selected_json_fields(PROTOCOL/'REQUESTS_LOCK.json',['stage_labels'])['stage_labels']
    def capture(stage):
        assert state.labels.tolist()==locked_stages[stage]['labels']
        assert state.versions.tolist()==locked_stages[stage]['versions']
        assert np.all(state.observed[~state.known]==0)
        pilot_mask=np.zeros(len(state.labels),bool);pilot_mask[state.pilot]=True
        fields=dict(proxy=state.proxy,observed=state.observed,labels=state.labels,versions=state.versions,
                    known=state.known,certainty=state.certainty,pilot=pilot_mask,pilot_rows=state.pilot,
                    Q_mean=state.q_mean,H_mean=state.h_mean,Q_sample=state.q_sample,H_sample=state.h_sample)
        for k,a in fields.items(): arrays[stage+'__'+k]=a.copy()
        heads[stage+'__mean']=state.w_mean.copy()
        heads[stage+'__paired025']=state.weights(.25,paired=True)
        heads[stage+'__sample_current']=state.w_sample.copy()
        receipt=dict(count=int(state.known.sum()),known_rows=np.flatnonzero(state.known).tolist(),
            certainty_rows=np.flatnonzero(state.certainty).tolist(),pilot_rows=state.pilot.tolist(),
            labels=semantic(state.labels),versions=semantic(state.versions),proxy=semantic(state.proxy),
            observed_exact=semantic(state.observed),sigma_selection=False,feature_version='v2')
        stage_receipts[stage]={name:dict(receipt) for name in METHODS}
        assert stage_receipts[stage]['mean']==stage_receipts[stage]['paired025']
        state_bytes[stage]=state.numeric_payload()
    capture('before_R1')
    for event in ['R1','R2']:
        if event=='R2':
            rows=np.arange(4000,5000,dtype=np.int64)
            state.append(provider.authorize(rows,'append_E'),np.asarray(p['initial_E_labels'],dtype=np.int64),feature_version='v2')
            capture('before_R2')
        # Provider interprets only request manifest; evaluator truth stays opaque.
        request=selected_json_fields(PROTOCOL/'REQUESTS_LOCK.json',['requests'])['requests'][event]
        rows=np.asarray(request['rows'],dtype=np.int64)
        assert len(rows)==p['request_counts'][event]
        old_known=state.known[rows].copy();fresh=rows[~old_known]
        entries=request['entries'];assert [q['row'] for q in entries]==rows.tolist()
        assert [q['global_id'] for q in entries]==ids[rows].tolist()
        assert all(bool(q['target_evidence_known_at_announcement'])==bool(k) for q,k in zip(entries,old_known))
        # All announced rows become authoritative, including known pilot rows.
        state.observe(rows,provider.authorize(rows,event+'_refresh'),feature_version='v2',authoritative=True)
        capture('after_refresh_'+event)
        release_log.append(dict(event=event,release_stage='after_refresh_'+event,
                                label_apply_stage='after_'+event,count=len(rows),new_label_used_before_refresh=False))
        state.relabel([Change(int(q['row']),int(q['expected_version']),int(q['old_label']),int(q['new_label']))
                       for q in entries],feature_version='v2')
        capture('after_'+event)
        events[event]=dict(requested_rows=rows.tolist(),fresh_rows=fresh.tolist(),
            already_known_rows=rows[old_known].tolist(),new_history_rows=fresh[fresh<4000].tolist(),
            global_ids=ids[rows].tolist(),fresh_global_ids=ids[fresh].tolist(),
            request_count=len(rows),new_target_count=len(fresh),sigma_selection=False)
    elapsed=time.perf_counter()-start
    np.savez_compressed(ROOT/'built/HEADS.npz',**heads)
    np.savez_compressed(ROOT/'built/STATES_ARRAYS.npz',**arrays)
    write(ROOT/'ACCESS_RECEIPTS.json',dict(task=TASK,status='PASS',stages=stage_receipts,events=events,
        initial=dict(rows=initial.tolist(),global_ids=ids[initial].tolist(),count=1024),
        append_E=dict(rows=list(range(4000,5000)),global_ids=ids[4000:5000].tolist(),count=1000),
        final_candidate_unique_target_rows=len(provider.allowed),
        final_candidate_rows=sorted(provider.allowed),final_candidate_global_ids=ids[sorted(provider.allowed)].tolist(),
        logical_access_log=provider.access_log,release_log=release_log,
        cache_physical_rows=6000,physical_encoding_is_not_logical_candidate_access=True,
        mean_paired_receipts_identical=True,eval_rows_given_to_candidate=False,sigma_used_for_selection=False))
    write(ROOT/'built/BUILD_COST.json',dict(task=TASK,candidate_build_wall_seconds=elapsed,
         state_array_bytes=state_bytes,initial_heads_count=3,total_stage_heads=18,
         performance_computed=False,full_reference_built=False,evaluation_truth_deserialized=False))
    names=['PUBLIC_PROTOCOL.json','PROTOCOL_GATE.json','FEATURE_CACHE_LOCK.json','FASTFILL_PROXY_LOCK.json',
           'DINO_IDENTITY_GATE.json','EXECUTION_SOURCE_LOCK.json','built/HEADS.npz','built/STATES_ARRAYS.npz',
           'ACCESS_RECEIPTS.json','built/BUILD_COST.json']
    write(ROOT/'BUILD_LOCK.json',dict(task=TASK,status='ALL_CANDIDATES_LOCKED_BEFORE_EVALUATION',
        stream_id=907001,methods=METHODS,stages=STAGES,files={n:asset(ROOT/n) for n in names},
        performance_computed=False,eval_truth_read=False,full_reference_built=False,
        protocol_rng_regenerated=False,created_utc=now()))
    print(json.dumps(dict(status='PASS',phase='build',heads=18,
        candidate_target_unique=len(provider.allowed),candidate_build_seconds=elapsed,performance_computed=False)))

def evaluate():
    p=gates();source_guard();lock=read(ROOT/'BUILD_LOCK.json')
    assert lock['status']=='ALL_CANDIDATES_LOCKED_BEFORE_EVALUATION'
    assert lock['performance_computed'] is False
    for rec in lock['files'].values():verify(rec)
    assert not (ROOT/'EVALUATION_LOCK.json').exists()
    # Exclusive marker: no modified scientific implementation may be rerun after this.
    write(ROOT/'evaluation/PERFORMANCE_ACCESS_STARTED.json',dict(task=TASK,created_utc=now(),
        build_lock_sha256=sha(ROOT/'BUILD_LOCK.json'),one_shot=True))
    _, _, solve, cross=imports_core();start=time.perf_counter()
    # First interpretation of evaluator truth in the scientific execution path.
    truth_lock=selected_json_fields(PROTOCOL/'REQUESTS_LOCK.json',['evaluation_truth'])['evaluation_truth']
    truth=np.asarray(truth_lock['labels'],dtype=np.int64)
    assert len(truth)==1000 and truth_lock['global_ids']==p['split']['groups']['evaluation']['global_ids']
    assert semantic(truth)['c_order_raw_sha256']==truth_lock['labels_sha256']
    feature=read(ROOT/'FEATURE_CACHE_LOCK.json')
    cache=np.load(verify(feature['assets']['B']),allow_pickle=False).astype(np.float64)
    b=aug(cache[:5000]);v=aug(cache[5000:])
    with np.load(ROOT/'built/HEADS.npz',allow_pickle=False) as z:heads={k:z[k] for k in z.files}
    with np.load(ROOT/'built/STATES_ARRAYS.npz',allow_pickle=False) as z:states={k:z[k] for k in z.files}
    scores={};predictions={};full={};stage_metrics=[];request_metrics=[]
    for stage in STAGES:
        labels=states[stage+'__labels'];n=len(labels)
        wf=solve(b[:n].T@b[:n]+np.eye(769),cross(b[:n],labels,100))
        full[stage+'__full_target_ridge']=wf;reference=v@wf
        for method in ALL_METHODS:
            w=wf if method=='full_target_ridge' else heads[stage+'__'+method]
            score=v@w;pred=score.argmax(1);key=stage+'__'+method
            scores[key]=score;predictions[key]=pred
            stage_metrics.append(dict(stage=stage,method=method,accuracy=float(np.mean(pred==truth)),
                score_rms=rms(score-reference),weight_rel=relative(w-wf,wf),
                known=int(states[stage+'__known'].sum())))
    for event in ['R1','R2']:
        ref=scores['after_'+event+'__full_target_ridge']-scores['after_refresh_'+event+'__full_target_ridge']
        for method in ALL_METHODS:
            delta=scores['after_'+event+'__'+method]-scores['after_refresh_'+event+'__'+method]
            total=scores['after_'+event+'__'+method]-scores['before_'+event+'__'+method]
            request_metrics.append(dict(event=event,method=method,request_count=p['request_counts'][event],
                response_error_rms=rms(delta-ref),reference_response_rms=rms(ref),response_rel=relative(delta-ref,ref),
                total_maintenance_error_rms=rms(total-ref),reference_total_action_rms=rms(ref),
                total_action_rel=relative(total-ref,ref)))
    primary={}
    for event in ['R1','R2']:
        rows={q['method']:q for q in request_metrics if q['event']==event}
        a=rows['mean']['response_rel'];z=rows['paired025']['response_rel']
        primary[event]=dict(mean_response_rel=a,paired025_response_rel=z,direction_pass=bool(z<a),
                            signed_relative_reduction=(a-z)/a if a!=0 else None,
                            signed_relative_reduction_defined=a!=0)
    count=sum(v['direction_pass'] for v in primary.values())
    status={2:'FINAL_UNSEEN_PRIMARY_CONFIRMED_BOTH',1:'FINAL_UNSEEN_PRIMARY_PARTIAL',
            0:'FINAL_UNSEEN_PRIMARY_NOT_CONFIRMED'}[count]
    write(ROOT/'PRIMARY_RESULT.json',dict(task=TASK,status=status,events=primary,stream_id=907001,
        stream_count=1,criterion='response_rel(paired025) < response_rel(mean), separately R1/R2',
        pass_threshold='direction only',secondary_used_to_redefine_primary=False,
        p_values_from_dependent_stage_cells=False,next_action='independent audit, package, STOP'))
    raw=np.load(verify(read(ROOT/'FASTFILL_PROXY_LOCK.json')['assets']['proxy']),allow_pickle=False).astype(np.float64)
    fit=np.asarray(p['fit_pilot']['fit']['rows'],dtype=np.int64);pilot=np.asarray(p['fit_pilot']['pilot']['rows'],dtype=np.int64)
    hold=np.ones(4000,bool);hold[fit]=False;hold[pilot]=False;hold_rows=np.flatnonzero(hold)
    assert len(hold_rows)==2976
    z=raw[hold];target=cache[:4000][hold];zc=z-z.mean(0);tc=target-target.mean(0)
    clean_history=states['after_R2__labels'][:4000]
    rr=retrieval(cache[5000:],z,truth,clean_history[hold])
    unique,counts=np.unique(z,axis=0,return_counts=True)
    m=dict(n=2976,feature_rms=rms(z-target),
        cosine_similarity=float(np.mean(np.sum(z*target,axis=1)/np.maximum(np.linalg.norm(z,axis=1)*np.linalg.norm(target,axis=1),1e-15))),
        centered_output_rms=rms(zc),centered_target_rms=rms(tc),
        centered_output_target_variation_ratio=rms(zc)/rms(tc),covariance_trace_ratio=float(np.sum(zc*zc)/np.sum(tc*tc)),
        exact_distinct_rows=len(unique),duplicate_rows=len(z)-len(unique),duplicate_groups=int(np.sum(counts>1)),
        largest_duplicate_group=int(counts.max()),retrieval_top1=rr['top1'],retrieval_map=rr['map'],
        queries_with_relevant=rr['queries_with_relevant'],
        before_R1_mean_accuracy=next(q['accuracy'] for q in stage_metrics if q['stage']=='before_R1' and q['method']=='mean'))
    retrieval_stages=[]
    for stage in ['before_R1','after_R2']:
        means=states[stage+'__proxy'][:4000,:-1].copy();known=states[stage+'__known'][:4000]
        means[known]=cache[:4000][known]
        retrieval_stages.append(dict(stage=stage,**retrieval(cache[5000:],means[hold],truth,clean_history[hold])))
    write(ROOT/'FRONTEND_QUALIFICATION.json',dict(task=TASK,status='SECONDARY_REPORT_ONLY',hold_rows=hold_rows.tolist(),
        hold_global_ids=np.asarray(p['split']['selected_ids'])[hold_rows].tolist(),metrics=m,
        retrieval=retrieval_stages,scope='eval1000 target queries -> history2976 class-label cosine gallery; frozen legacy bridge helper',
        no_post_normalization=True,no_pilot_bias=True,no_sigma_evidence_selection=True,not_used_to_change_method=True))
    writecsv(ROOT/'FRONTEND_QUALIFICATION.csv',[m]);writecsv(ROOT/'STAGE_METRICS.csv',stage_metrics)
    writecsv(ROOT/'EVENTS_ALL.csv',stage_metrics);writecsv(ROOT/'REQUESTS_ALL.csv',request_metrics)
    np.savez_compressed(ROOT/'evaluation/HEADS_FULL.npz',**full)
    np.savez_compressed(ROOT/'evaluation/SCORES.npz',**scores)
    np.savez_compressed(ROOT/'evaluation/PREDICTIONS.npz',**predictions)
    elapsed=time.perf_counter()-start
    write(ROOT/'COST_ACCOUNTING.json',dict(task=TASK,
        DINO_encoding=read(ROOT/'FEATURE_CACHE_LOCK.json')['costs'],
        FastFill_forward=read(ROOT/'FASTFILL_PROXY_LOCK.json').get('cost',{}),
        candidate_build=read(ROOT/'built/BUILD_COST.json'),evaluation_wall_seconds=elapsed,
        frontend_parameter_bytes=read(ROOT/'FASTFILL_PROXY_LOCK.json').get('parameter_bytes'),
        frontend_checkpoint_bytes=60189229,candidate_unique_target_B_rows=read(ROOT/'ACCESS_RECEIPTS.json')['final_candidate_unique_target_rows'],
        physical_B_cache_rows=6000,physical_cache_distinct_from_candidate_access=True,
        upstream_training='static prior cost reference; not incurred again this task',
        static_upstream_costs=read(ROOT/'provenance/STATIC_UPSTREAM_COSTS.json'),
        upstream_cost_sources=['fastfill_dino_normalized_adaptation_02/EPOCH_METRICS.json',
                              'fastfill_dino_transformation_train_01/FASTFILL_TRANSFORMATION_LOCK.json'],
        resource_ranking_reopened=False))
    write(ROOT/'evaluation/METRIC_DEFINITION.json',dict(task=TASK,
        pure_reference='full_after_Rk - full_after_refresh_Rk',
        response_error='(candidate_after_Rk - candidate_after_refresh_Rk) - pure_reference',
        total_maintenance_error='(candidate_after_Rk - candidate_before_Rk) - pure_reference',
        reference_total_action_rms='RMS(pure_reference); exact legacy bridge denominator',
        relative_denominator_floor=1e-15,signed_primary_reduction='(mean-paired)/mean; mean=0 is undefined',
        source='fastfill_common_evidence_revision_bridge_01/code/bridge_fastfill.py:364-383'))
    names=['PRIMARY_RESULT.json','FRONTEND_QUALIFICATION.json','FRONTEND_QUALIFICATION.csv',
           'STAGE_METRICS.csv','EVENTS_ALL.csv','REQUESTS_ALL.csv','COST_ACCOUNTING.json',
           'evaluation/HEADS_FULL.npz','evaluation/SCORES.npz','evaluation/PREDICTIONS.npz',
           'evaluation/PERFORMANCE_ACCESS_STARTED.json','evaluation/METRIC_DEFINITION.json']
    write(ROOT/'EVALUATION_LOCK.json',dict(task=TASK,status='COMPLETE_ONE_SHOT',files={n:asset(ROOT/n) for n in names},
        build_lock_sha256=sha(ROOT/'BUILD_LOCK.json'),source_lock_sha256=sha(ROOT/'EXECUTION_SOURCE_LOCK.json'),
        primary_status=status,created_utc=now(),official_TEST_used=False,stream_count=1,
        protocol_rng_regenerated=False,no_method_change_after_performance=True))
    print(json.dumps(dict(status='PASS',phase='evaluate',primary_status=status,events=primary,
                         evaluation_seconds=elapsed,next_action='audit, package, STOP')))

if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['build','evaluate']
    {'build':build,'evaluate':evaluate}[sys.argv[1]]()
