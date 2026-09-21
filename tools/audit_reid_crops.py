#!/usr/bin/env python3
"""Offline, read-only crop audit. Never imports the car main or opens serial/camera.

Default mode reads saved evidence. --infer explicitly runs the configured ReID
model on saved crops and refuses to start while the car entrypoint is running.
An optional same-weight ONNX reference uses the identical preprocessed tensors.
"""
from __future__ import annotations
import argparse
import configparser
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))


def cap_list(value):
    return [int(v) for v in value.split(",") if v.strip()]


def car_processes(proc=Path("/proc")):
    found=[]
    for path in proc.glob("[0-9]*/cmdline"):
        try:
            args=path.read_bytes().decode(errors="replace").split("\0")
        except OSError:
            continue
        if any(Path(arg).name in {"request_0513_modular.py","run_request_0428_modular.sh"}
               for arg in args if arg):
            found.append(int(path.parent.name))
    return found


def evidence_rows(run_dir, caps):
    result=[]
    with (run_dir/"request_0513_modular.log").open() as stream:
        for line in stream:
            if "reid_match_evidence {" not in line:
                continue
            event=json.loads(line.split("reid_match_evidence ",1)[1])
            cap=event.get("query_metadata",{}).get("capture_frame_id")
            if cap not in caps:
                continue
            match=event.get("match_evidence") or {}
            winner=match.get("winner") or {}
            result.append(dict(cap=cap,track=event["track_id"],uid=event["output_uid"],
                query_metadata=event.get("query_metadata"),frame_index=event["frame_index"],
                reason=event["reason"],logged_fused_distance=match.get("distance"),
                template_cap=winner.get("metadata",{}).get("capture_frame_id"),
                recent_full=event.get("template_recent_evidence"),
                recent_partial=event.get("template_recent_partial_evidence"),
                logged_partial_distance=event.get("partial_distance"),
                geometry=event.get("reacquire_geometry"),bank_updated=event["bank_updated"]))
    return result


def crop_path(run_dir,cap):
    paths=sorted((run_dir/"reid_diagnostics").glob(f"*_capture_{cap:08d}_track_*.png"))
    if len(paths)!=1:
        raise ValueError(f"CAP{cap}: expected one saved crop, found {len(paths)}; ambiguous crops cannot be silently selected")
    return paths[0]


def cosine(a,b):
    import numpy as np
    if a is None or b is None:
        return None
    a=np.asarray(a).reshape(-1);b=np.asarray(b).reshape(-1)
    if a.shape!=b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("invalid or incompatible embedding")
    norms=float(np.linalg.norm(a)*np.linalg.norm(b))
    if norms<=1e-12:
        raise ValueError("zero embedding")
    return float(1-a.dot(b)/norms)


def model_config(config_path):
    from rk_vision.reid import OSNetConfig
    cfg=configparser.ConfigParser(interpolation=None);cfg.read(config_path)
    v=cfg["vision"]
    return OSNetConfig(model_path=str(ROOT/v["reid_model_path"]),
        input_width=v.getint("reid_input_width",128),input_height=v.getint("reid_input_height",256),
        input_format=v.get("reid_input_format","RGB"),input_dtype=v.get("reid_input_dtype","float32"),
        input_layout=v.get("reid_input_layout","NCHW"),normalize=v.get("reid_normalize","imagenet"),
        backend=v.get("rknn_backend","auto"),color_fusion_enable=False,
        partial_osnet_enable=True,partial_appearance_enable=True,
        partial_torso_top_ratio=v.getfloat("reid_partial_torso_top_ratio",.10),
        partial_torso_bottom_ratio=v.getfloat("reid_partial_torso_bottom_ratio",.86),
        partial_torso_side_ratio=v.getfloat("reid_partial_torso_side_ratio",.10)),v.getfloat("reid_color_fusion_weight",.35)


def extract_saved(run_dir,caps,config):
    import cv2
    from rk_vision.reid import OSNetRKNNExtractor,_color_signature,_fuse_appearance_features
    from rk_vision.yolo11 import Detection
    cfg,weight=config
    encoder=OSNetRKNNExtractor(cfg)
    vectors={};provenance=[]
    try:
        for cap in sorted(set(caps)):
            active=car_processes()
            if active:
                raise RuntimeError(f"car runtime active: {active}; stop offline inference")
            path=crop_path(run_dir,cap);crop=cv2.imread(str(path))
            if crop is None:
                raise ValueError(f"unreadable crop: {path}")
            h,w=crop.shape[:2]
            raw=encoder.extract(crop,[Detection((0,0,w,h),1.,0)],"BGR")[0]
            if raw is None:
                raise ValueError(f"no feature for CAP{cap}")
            color=_color_signature(crop)
            vectors[cap]={"osnet":raw,"color":color,
                "fused":_fuse_appearance_features(raw,color,weight),
                "torso":encoder.last_partial_features[0]}
            provenance.append(dict(cap=cap,path=str(path),width=w,height=h,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                raw_feature_dimensions=int(raw.size)))
    finally:
        encoder.release()
    return vectors,provenance


def replay_policy(vectors, events, template_caps, query_caps, config_path):
    """Selected-gallery replay, not reconstruction of the whole run's bank."""
    from dataclasses import fields
    from rk_vision.identity_bank import IdentityBank, IdentityBankConfig
    parser=configparser.ConfigParser(interpolation=None);parser.read(config_path)
    options={};defaults=IdentityBankConfig()
    for f in fields(defaults):
        if parser.has_option("identity_bank",f.name):
            default=getattr(defaults,f.name)
            getter=parser.getboolean if isinstance(default,bool) else parser.getint if isinstance(default,int) else parser.getfloat
            options[f.name]=getter("identity_bank",f.name)
    by_cap={e["cap"]:e for e in events}
    result={}
    for enabled in (False,True):
        cfg=IdentityBankConfig(**{**options,"template_memory_enable":True,"template_crosscheck_enable":enabled})
        bank=IdentityBank(cfg)
        for cap in sorted(template_caps):
            event=by_cap[cap];m=event["query_metadata"];v=vectors[cap]
            if not bank.identities:
                bank._create_identity(v["fused"],event["frame_index"],m,v["torso"])
            else:
                entry=bank.identities[1]
                entry.add(v["fused"],event["frame_index"],cfg.max_features,metadata=m)
                entry.add_partial(v["torso"],event["frame_index"],cfg.partial_max_features,metadata=m)
        first=by_cap[query_caps[0]]
        reference=first.get("geometry",{}).get("reference")
        if not reference:
            raise ValueError("replay needs the logged last trusted geometry")
        bank.identities[1].last_strong_observation=dict(reference)
        bank.track_to_uid[int(reference["track_id"])]=1
        bank.track_last_seen_frame[int(reference["track_id"])]=int(reference["frame_index"])
        bank.identities[1].last_seen_frame=int(reference["frame_index"])
        rows=[]
        for cap in query_caps:
            event=by_cap[cap];m=dict(event["query_metadata"]);v=vectors[cap]
            # These are unconfirmed search candidates in both counterfactuals.
            # Keep capture times, yaw and detector boxes; do not replay motors.
            m["search_reacquire_context_active"]=True
            uid=bank.assign(track_id=event["track"],feature=v["fused"],partial_feature=v["torso"],
                confidence=m["detector_confidence"],area=m["detector_area_ratio"]*640*480,
                frame_index=event["frame_index"],candidate_count=int(m.get("candidate_count",1)),
                bbox_quality_ok=bool(m.get("quality_bbox_ok")),bbox_quality_tier=m.get("bbox_quality_tier"),
                sample_metadata=m,preferred_uid=1,preferred_candidate_ok=m.get("search_direction_compatible") is not False)
            a=bank.last_assignments[event["track"]]
            rows.append(dict(cap=cap,uid=uid,reason=a["reason"],partial_state=a.get("reacquire_partial_state"),
                recent_full=a.get("template_recent_evidence"),recent_partial=a.get("reacquire_recent_partial_evidence")))
        result["crosscheck_on" if enabled else "crosscheck_off"]=rows
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir",type=Path,required=True)
    parser.add_argument("--templates",type=cap_list,required=True)
    parser.add_argument("--queries",type=cap_list,required=True)
    parser.add_argument("--positive-caps",type=cap_list,default=[])
    parser.add_argument("--negative-caps",type=cap_list,default=[])
    parser.add_argument("--config",type=Path,default=ROOT/"car_control_modular/config/reid_runtime.ini")
    parser.add_argument("--infer",action="store_true")
    parser.add_argument("--reference-onnx",type=Path)
    parser.add_argument("--replay-policy",action="store_true",help="selected-gallery counterfactual; requires --infer")
    parser.add_argument("--feature-output",type=Path,help="optional new .npz archive of offline vectors")
    parser.add_argument("--output",type=Path)
    args=parser.parse_args(argv)
    if set(args.positive_caps)&set(args.negative_caps):
        parser.error("positive and negative labels must be disjoint")
    if args.output and args.output.exists():
        parser.error("output already exists; refuse overwrite")
    if args.feature_output and args.feature_output.exists():
        parser.error("feature output already exists; refuse overwrite")
    if (args.feature_output or args.replay_policy) and not args.infer:
        parser.error("feature export and policy replay require --infer")
    if args.reference_onnx and (not args.infer or not args.reference_onnx.is_file()):
        parser.error("reference requires --infer and an existing same-weight ONNX file")
    report=dict(mode="offline_inference" if args.infer else "log_only",
        log_evidence=evidence_rows(args.run_dir,set(args.queries+args.templates)),
        reference_status="not_supplied",labels={str(c):"correct" for c in args.positive_caps})
    report["labels"].update({str(c):"wrong" for c in args.negative_caps})
    if args.infer:
        if car_processes():
            parser.error("car runtime active; offline inference is not allowed")
        config=model_config(args.config)
        vectors,provenance=extract_saved(args.run_dir,args.templates+args.queries,config)
        report["crops"]=provenance
        cfg,weight=config
        report["model"]=dict(path=cfg.model_path,sha256=hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest(),
            width=cfg.input_width,height=cfg.input_height,format=cfg.input_format,dtype=cfg.input_dtype,
            layout=cfg.input_layout,normalize=cfg.normalize,color_weight=weight)
        report["pairs"]=[dict(query=q,template=t,**{
            name:cosine(vectors[q][name],vectors[t][name]) for name in ("osnet","color","fused","torso")})
            for q in args.queries for t in args.templates if q!=t]
        if args.feature_output:
            import numpy as np
            args.feature_output.parent.mkdir(parents=True,exist_ok=True)
            with args.feature_output.open("xb") as stream:
                np.savez_compressed(stream,**{f"cap_{c}_{name}":v for c,items in vectors.items()
                                              for name,v in items.items() if v is not None})
            report["feature_archive"]=str(args.feature_output)
        if args.replay_policy:
            query_caps=[c for c in args.queries if c in args.negative_caps]
            if not query_caps:
                parser.error("policy replay needs explicitly labeled negative CAPs")
            report["selected_gallery_replay"]=replay_policy(vectors,report["log_evidence"],
                args.templates,query_caps,args.config)
            report["replay_scope"]="selected saved templates, actual crops/times/boxes; forced search context; not full-run state reconstruction"
        report["threshold_summary"]=[]
        for label in ("correct","wrong"):
            for threshold in (.15,.20,.30):
                distances=[min(p["fused"] for p in report["pairs"] if p["query"]==q)
                    for q in args.queries if report["labels"].get(str(q))==label
                    and any(p["query"]==q for p in report["pairs"])]
                report["threshold_summary"].append(dict(label=label,threshold=threshold,
                    samples=len(distances),appearance_passes=sum(d<=threshold for d in distances)))
        if args.reference_onnx:
            reference,_=extract_saved(args.run_dir,args.templates+args.queries,
                (replace(cfg,model_path=str(args.reference_onnx),backend="onnxruntime"),weight))
            report["reference_status"]="compared_same_preprocessing_user_supplied_weights"
            report["reference_model_sha256"]=hashlib.sha256(args.reference_onnx.read_bytes()).hexdigest()
            report["reference_embedding_distance"]=[dict(cap=c,
                osnet=cosine(vectors[c]["osnet"],reference[c]["osnet"]),
                torso=cosine(vectors[c]["torso"],reference[c]["torso"])) for c in sorted(vectors)]
    payload=json.dumps(report,ensure_ascii=False,allow_nan=False,indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True,exist_ok=True)
        with args.output.open("x") as stream:
            stream.write(payload+"\n")
        print(f"report={args.output}")
    else:
        print(payload)
    return 0


if __name__=="__main__":
    raise SystemExit(main())
