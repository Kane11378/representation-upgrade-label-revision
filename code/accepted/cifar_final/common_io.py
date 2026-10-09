"""Local metadata/serialization utilities. Never create a protocol with RNG."""
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
WORK = ROOT.parent
PROTOCOL = WORK / 'fastfill_final_unseen_protocol_freeze_01'
TASK = 'FASTFILL_FINAL_UNSEEN_CONFIRMATION_01'
STAGES = ['before_R1', 'after_refresh_R1', 'after_R1', 'before_R2', 'after_refresh_R2', 'after_R2']
METHODS = ['mean', 'paired025', 'sample_current']
ALL_METHODS = METHODS + ['full_target_ridge']

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024*1024), b''): h.update(b)
    return h.hexdigest()

def semantic(a):
    a = np.ascontiguousarray(a)
    return dict(dtype=str(a.dtype), shape=list(a.shape), c_order_raw_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest())

def asset(path, a=None):
    p = Path(path).resolve()
    v = dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p))
    if a is not None: v['semantic'] = semantic(a)
    return v

def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))

def write(path, value):
    path = Path(path)
    assert path.resolve().is_relative_to(ROOT.resolve())
    with path.open('x', encoding='utf-8', newline='\n') as f:
        json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False); f.write('\n')

def verify(record):
    p = Path(record['path'])
    assert p.stat().st_size == record['bytes'] and sha(p) == record['sha256'], str(p)
    return p

def source_guard():
    lock = read(ROOT/'EXECUTION_SOURCE_LOCK.json')
    assert lock['status'] == 'FROZEN_BEFORE_EXECUTION'
    for n, rec in lock['files'].items():
        assert (ROOT/n).resolve() == Path(rec['path']).resolve()
        verify(rec)
    gate = read(ROOT/'PROTOCOL_GATE.json')
    assert gate['status'] == 'PASS'
    verify(gate['protocol_zip'])
    for rec in gate['protocol_assets'].values(): verify(rec)
    verify(gate['public_protocol'])
    return lock

def selected_json_fields(path, names):
    """Decode selected top-level values, skipping other JSON values as opaque text.

    In particular, provider/build cannot deserialize the evaluation_truth value.
    Identity hashing may read its bytes; evaluator interprets it only after lock.
    """
    text = Path(path).read_text(encoding='utf-8')
    decoder = json.JSONDecoder()
    names = set(names); result = {}; i = 0
    def ws(p):
        while p < len(text) and text[p].isspace(): p += 1
        return p
    def end(p):
        p = ws(p)
        if text[p] == '"': return decoder.raw_decode(text, p)[1]
        if text[p] in '[{':
            stack = [text[p]]; p += 1; quoted = False; escaped = False
            while stack:
                c = text[p]
                if quoted:
                    if escaped: escaped = False
                    elif c == '\\': escaped = True
                    elif c == '"': quoted = False
                elif c == '"': quoted = True
                elif c in '[{': stack.append(c)
                elif c in ']}':
                    assert (stack.pop(), c) in [('[', ']'), ('{', '}')]
                p += 1
            return p
        while p < len(text) and text[p] not in ',}': p += 1
        return p
    i = ws(i); assert text[i] == '{'; i += 1
    while True:
        i = ws(i)
        if text[i] == '}': break
        key, i = decoder.raw_decode(text, i); i = ws(i); assert text[i] == ':'; i = ws(i+1)
        stop = end(i)
        if key in names: result[key] = json.loads(text[i:stop])
        i = ws(stop)
        if text[i] == ',': i += 1
        else: assert text[i] == '}'; break
    assert set(result) == names, names-set(result)
    return result

