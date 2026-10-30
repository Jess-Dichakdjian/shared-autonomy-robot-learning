"""Evidence-only benchmark summary, interval, latency and representative videos."""
import argparse
import csv
import datetime
import json
from pathlib import Path
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from thesis_vla.benchmark import atomic_json,completed_episodes,wilson


def report(output):
    output=Path(output)
    records=completed_episodes(output)
    if not records:
        raise ValueError("No complete episodes")
    latencies=[]; episode_rows=[]
    for r in records:
        path=output/"episodes"/f"episode_{r['episode_id']:03d}"
        ep=json.loads((path/"episodes.jsonl").read_text())
        rows=[json.loads(s) for s in (path/"steps.jsonl").read_text().splitlines()]
        times=[row["inference_seconds"] for row in rows if row["chunk_start"]]
        latencies.extend(times)
        episode_rows.append({"episode_id":r["episode_id"],"seed":r["seed"],
            "saved_init_state_index":r["saved_init_state_index"],"success":r["success"],"steps":r["steps"],
            "termination_reason":r["termination_reason"],"chunk_predictions":len(times),
            "mean_chunk_seconds":float(np.mean(times)),
            "video":str(path/ep["video"]) if ep["video"] else None})
    successes=sum(r["success"] for r in records)
    reasons={k:sum(r["termination_reason"]==k for r in records) for k in sorted({r["termination_reason"] for r in records})}
    representatives={}
    for outcome,name in ((True,"success"),(False,"failure")):
        group=sorted([r for r in episode_rows if r["success"]==outcome],key=lambda r:r["steps"])
        representatives[name]=group[len(group)//2] if group else None
    memory={}
    sample_path=output/"gpu_samples.csv"
    if sample_path.exists():
        samples=list(csv.DictReader(sample_path.open()))
        included=[]
        for r in samples:
            stamp=datetime.datetime.strptime(r["timestamp"].strip(),"%Y/%m/%d %H:%M:%S.%f").replace(tzinfo=datetime.timezone.utc).timestamp()
            if any(ep["attempt_started_at"]<=stamp<=ep["attempt_ended_at"] for ep in records):
                included.append(r)
        for index in sorted({r[" index"].strip() for r in included}):
            rows=[r for r in included if r[" index"].strip()==index]
            memory[index]={"sampled_peak_total_used_mib":max(int(r[" memory.used [MiB]"].strip().split()[0]) for r in rows),
                "minimum_free_mib":min(int(r[" memory.free [MiB]"].strip().split()[0]) for r in rows)}
    allocator={}
    for r in records:
        for device,values in r.get("gpu_allocator_peaks",{}).items():
            allocator.setdefault(device,{"allocated":0,"reserved":0})
            for key in ("allocated","reserved"):
                allocator[device][key]=max(allocator[device][key],values[key])
    result={"evaluated_episodes":len(records),"successes":successes,"failures":len(records)-successes,
        "success_rate":successes/len(records),"wilson_95":wilson(successes,len(records)),
        "interval_scope":"Descriptive binomial interval for this fixed single-task scene/seed batch; not a cross-task or deployment guarantee.",
        "steps_total":sum(r["steps"] for r in records),"steps_mean":float(np.mean([r["steps"] for r in records])),
        "steps_min":min(r["steps"] for r in records),"steps_max":max(r["steps"] for r in records),
        "termination_reasons":reasons,"prediction_points":len(latencies),
        "chunk_latency_seconds":{"mean":float(np.mean(latencies)),"median":float(np.median(latencies)),
            "p95":float(np.quantile(latencies,.95)),"maximum":max(latencies)},
        "gpu_samples":memory,"gpu_allocator_peaks_bytes":allocator,
        "representative_videos":representatives,"episodes":episode_rows,
        "gp_dataset":json.loads((output/"dataset/schema.json").read_text()) if (output/"dataset/schema.json").exists() else None}
    atomic_json(output/"report.json",result)
    with (output/"episode_summary.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=list(episode_rows[0]));writer.writeheader();writer.writerows(episode_rows)
    return result


if __name__ == "__main__":
    p=argparse.ArgumentParser();p.add_argument("experiment",type=Path);args=p.parse_args()
    r=report(args.experiment)
    print(json.dumps({k:v for k,v in r.items() if k not in ("episodes","gp_dataset")},indent=2))
