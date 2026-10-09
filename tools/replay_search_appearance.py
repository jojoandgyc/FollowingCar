#!/usr/bin/env python3
"""Explicit offline saved-crop inference and selected-gallery policy comparison.

Never opens camera/serial or imports the car entrypoint. This is NOT a complete
run replay: the caller explicitly selects trusted templates and the reference.
"""
import argparse
import configparser
from dataclasses import fields, replace
import json
import logging
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.audit_reid_crops import cap_list, car_processes, extract_saved, model_config
from rk_vision.identity_bank import IdentityBank, IdentityBankConfig, _geometry_observation


def replay(events, vectors, templates, queries, reference, config):
    by={e['capture_frame_id']:e for e in events}
    result={}
    for threshold in (.34,.45):
        b=IdentityBank(replace(config,template_memory_enable=True,template_crosscheck_enable=True,
                               partial_match_threshold=threshold))
        for cap in sorted(templates):
            e=by[cap];m=dict(e['sample_metadata']);v=vectors[cap];f=e['frame_index']
            if not b.identities:
                b._create_identity(v['fused'],f,m,v['torso'])
            else:
                entry=b.identities[1]
                entry.add(v['fused'],f,config.max_features,metadata=m)
                entry.add_partial(v['torso'],f,config.partial_max_features,metadata=m)
        ref=by[reference]
        b.identities[1].last_strong_observation=_geometry_observation(
            dict(ref['sample_metadata'],track_id=ref['raw_track_id']),ref['frame_index'])
        output=[]
        for cap in sorted(queries):
            e=by[cap];m=dict(e['sample_metadata']);v=vectors[cap];track=e['raw_track_id']
            box=e['detector_bbox']
            uid=b.assign(track_id=track,feature=v['fused'],partial_feature=v['torso'],
                confidence=m['detector_confidence'],area=(box[2]-box[0])*(box[3]-box[1]),
                frame_index=e['frame_index'],candidate_count=int(m.get('candidate_count',1)),
                bbox_quality_ok=m['quality_bbox_ok'],bbox_quality_tier=m['bbox_quality_tier'],
                sample_metadata=m,preferred_uid=1,preferred_candidate_ok=m.get('search_direction_compatible') is not False)
            a=b.last_assignments[track]
            output.append(dict(cap=cap,uid=uid,reason=a['reason'],bank_updated=a.get('bank_updated'),
                partial_state=a.get('reacquire_partial_state'),observation=a.get('candidate_observation')))
        result[str(threshold)]=output
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--templates',type=cap_list,required=True)
    p.add_argument('--queries',type=cap_list,required=True)
    p.add_argument('--reference-cap',type=int,required=True)
    p.add_argument('--config',type=Path,default=ROOT/'car_control_modular/config/reid_runtime.ini')
    p.add_argument('--infer',action='store_true',help='required to use NPU on saved crops')
    a=p.parse_args()
    if not a.infer:p.error('explicit --infer required; no cameras/motors are used')
    if car_processes():p.error('car runtime is active; refuse offline inference')
    if set(a.templates)&set(a.queries):p.error('queries cannot bootstrap their own gallery')
    events=[json.loads(s) for s in (a.run_dir/'reid_diagnostics/events.jsonl').open()]
    selected=set(a.templates+a.queries+[a.reference_cap])
    for cap in selected:
        if sum(e['capture_frame_id']==cap for e in events)!=1:
            p.error(f'CAP{cap}: missing or ambiguous observation')
    parser=configparser.ConfigParser(interpolation=None);parser.read(a.config)
    options={};defaults=IdentityBankConfig()
    for f in fields(defaults):
        if not parser.has_option('identity_bank',f.name):continue
        d=getattr(defaults,f.name)
        getter=parser.getboolean if isinstance(d,bool) else parser.getint if isinstance(d,int) else parser.getfloat
        options[f.name]=getter('identity_bank',f.name)
    logging.disable(logging.CRITICAL)
    vectors,provenance=extract_saved(a.run_dir,a.templates+a.queries,model_config(a.config))
    print(json.dumps(dict(mode='selected_gallery_offline_inference',negative_crop_validation=False,
        templates=a.templates,reference_cap=a.reference_cap,crops=provenance,
        results=replay(events,vectors,a.templates,a.queries,a.reference_cap,IdentityBankConfig(**options))),
        ensure_ascii=False,indent=2))


if __name__=='__main__':main()
