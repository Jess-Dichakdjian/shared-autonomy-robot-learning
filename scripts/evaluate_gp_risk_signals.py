"""Evaluate frozen GP signals on the predeclared held-out BEAST episodes."""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr, bootstrap
from sklearn.metrics import roc_auc_score, average_precision_score

Z90, Z95 = 1.6448536269514722, 1.959963984540054


def read_episodes(root, task_id):
    result=[]
    for epdir in sorted((root/"episodes").glob("episode_*")):
        if not (epdir/"complete.json").exists(): continue
        episode=json.loads((epdir/"episodes.jsonl").read_text().splitlines()[0])
        if episode["success"] is None: raise ValueError("Unknown LIBERO success outcome")
        rows=[json.loads(line) for line in (epdir/"steps.jsonl").read_text().splitlines()]
        starts=[row for row in rows if row["chunk_start"]]
        X=[];Y=[];M=[];S=[]
        for row in starts:
            mon=row["passive_gp_prediction"]
            state=np.asarray(row["prediction_state"]["vla_normalized"],dtype=float)
            if not np.array_equal(state,np.asarray(mon["input_state_vla_normalized"],dtype=float)):
                raise ValueError("GP input state mismatch")
            X.append(state);Y.append(row["predicted_action"])
            M.append(mon["mean_raw_normalized_action"]);S.append(mon["predictive_std_raw_normalized_action"])
        result.append({
            "task_id":task_id,"state_index":episode["reset_info"]["saved_init_state_index"],
            "seed":episode["seed"],"success":bool(episode["success"]),"steps":episode["steps"],
            "termination_reason":episode["termination_reason"],"task":episode["task"],
            "episode_key":f"{task_id}:{episode['reset_info']['saved_init_state_index']}",
            "X":np.asarray(X,dtype=float),"Y":np.asarray(Y,dtype=float),
            "mean":np.asarray(M,dtype=float),"std":np.asarray(S,dtype=float),
            "chunk_steps":np.asarray([row["step"] for row in starts],dtype=int)
        })
    return result


def cluster_bootstrap_spearman(x,y,groups,seed=4901,repeats=2000):
    unique=np.unique(groups)
    if len(unique)<3:return {"estimate":None,"ci95":[None,None],"valid_replicates":0}
    estimate=float(spearmanr(x,y).statistic)
    rng=np.random.default_rng(seed); vals=[]
    group_indices={g:np.flatnonzero(groups==g) for g in unique}
    for _ in range(repeats):
        sampled=rng.choice(unique,size=len(unique),replace=True)
        ids=np.concatenate([group_indices[g] for g in sampled])
        if np.ptp(x[ids])>0 and np.ptp(y[ids])>0:
            v=float(spearmanr(x[ids],y[ids]).statistic)
            if np.isfinite(v):vals.append(v)
    ci=np.quantile(vals,[.025,.975]).tolist() if vals else [None,None]
    return {"estimate":estimate,"ci95":[float(v) for v in ci],"valid_replicates":len(vals),"resamples":repeats}


def safe_rank_metrics(labels,scores):
    if len(np.unique(labels))<2:
        return {"auroc":None,"average_precision":None,"status":"not evaluable: one outcome class"}
    return {"auroc":float(roc_auc_score(labels,scores)),
            "average_precision":float(average_precision_score(labels,scores)),
            "status":"exploratory: one adverse episode; estimates highly unstable" if int(np.sum(labels))<3 else "computed"}



def matched_random_budget(labels, episode_index_per_row, budget, *, seed=20261104, repeats=10000):
    """Randomly place exactly the observed request count over available chunk boundaries."""
    rng=np.random.default_rng(seed); n_rows=len(episode_index_per_row); n_eps=len(labels)
    failure_index=np.flatnonzero(labels==1)
    if len(failure_index)!=1:
        return {"status":"descriptive matched-budget simulation","budget":int(budget),"repeats":repeats}
    aucs=[];aps=[];detected=[];success_flagged=[]
    for _ in range(repeats):
        chosen=rng.choice(n_rows,size=budget,replace=False) if budget else np.asarray([],dtype=int)
        counts=np.bincount(episode_index_per_row[chosen],minlength=n_eps)
        metrics=safe_rank_metrics(labels,counts)
        aucs.append(metrics["auroc"]);aps.append(metrics["average_precision"])
        detected.append(bool(counts[failure_index[0]]>0))
        success_flagged.append(int(np.sum((counts>0)&(labels==0))))
    return {"status":"Monte Carlo; exact total request budget matched",
        "budget":int(budget),"available_chunk_boundaries":n_rows,"repeats":repeats,"seed":seed,
        "failure_episode_detection_probability":float(np.mean(detected)),
        "success_episodes_flagged_mean":float(np.mean(success_flagged)),
        "episode_count_score_auroc_mean":float(np.mean(aucs)),
        "episode_count_score_auroc_q025_q975":np.quantile(aucs,[.025,.975]).astype(float).tolist(),
        "episode_count_score_average_precision_mean":float(np.mean(aps)),
        "episode_count_score_average_precision_q025_q975":np.quantile(aps,[.025,.975]).astype(float).tolist(),
        "bias_note":"Uniformly sampling chunk boundaries gives longer episodes more opportunities to receive random requests."}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--model",type=Path,required=True)
    ap.add_argument("--task1",type=Path,required=True)
    ap.add_argument("--task4",type=Path,required=True)
    ap.add_argument("--threshold-config",type=Path,required=True)
    ap.add_argument("--output",type=Path,required=True)
    args=ap.parse_args()
    output=args.output;output.mkdir(parents=True,exist_ok=False)
    config=json.loads(args.threshold_config.read_text())
    bundle=joblib.load(args.model);gp=bundle["gp"]
    groups=read_episodes(args.task1,1)+read_episodes(args.task4,4)
    episodes=[e for e in groups if e["state_index"] in range(2,7)]
    if len(episodes)!=10:raise ValueError(f"Expected exactly 10 held-out episodes; got {len(episodes)}")
    X=np.concatenate([e["X"] for e in episodes]);Y=np.concatenate([e["Y"] for e in episodes])
    M=np.concatenate([e["mean"] for e in episodes]);S=np.concatenate([e["std"] for e in episodes])
    ep_keys=np.concatenate([[e["episode_key"]]*len(e["X"]) for e in episodes])
    task_ids=np.concatenate([[e["task_id"]]*len(e["X"]) for e in episodes])
    scales=gp.y_scaler.scale_
    U=np.sqrt(np.mean(np.square(S/scales[None,:]),axis=1))
    D=np.sqrt(np.mean(np.square((M-Y)/scales[None,:]),axis=1))
    sx=gp.x_scaler.transform(X);train_scaled=gp.models[0].X_train_
    H=np.linalg.norm(sx[:,None,:]-train_scaled[None,:,:],axis=2).min(axis=1)
    err=np.sqrt(np.mean(np.square((Y-M)/scales[None,:]),axis=1))
    labels=np.concatenate([[int(not e["success"])]*len(e["X"]) for e in episodes])

    gp_mae=np.mean(np.abs(Y-M),axis=0);gp_rmse=np.sqrt(np.mean((Y-M)**2,axis=0))
    ridge_mean=bundle["ridge"].predict(X);ridge_mae=np.mean(np.abs(Y-ridge_mean),axis=0)
    ridge_rmse=np.sqrt(np.mean((Y-ridge_mean)**2,axis=0))
    cov90=np.mean(np.abs(Y-M)<=Z90*S,axis=0);cov95=np.mean(np.abs(Y-M)<=Z95*S,axis=0)
    multiplier=float(config["validation"]["descriptive_calibration_multiplier"])
    cal_cov90=np.mean(np.abs(Y-M)<=Z90*S*multiplier,axis=0)
    cal_cov95=np.mean(np.abs(Y-M)<=Z95*S*multiplier,axis=0)

    thresholds=config["request_thresholds_from_validation_only"]
    signal_map={"gp_uncertainty":U,"gp_vla_disagreement":D,"state_novelty":H}
    threshold_map={"gp_uncertainty":thresholds["gp_uncertainty_standardized_rms_90th"],
                   "gp_vla_disagreement":thresholds["gp_vla_disagreement_standardized_rms_90th"],
                   "state_novelty":thresholds["nearest_training_state_distance_90th"]}
    episode_rows=[]
    for e in episodes:
        ids=np.flatnonzero(ep_keys==e["episode_key"])
        item={"episode_key":e["episode_key"],"task_id":e["task_id"],"state_index":e["state_index"],
              "seed":e["seed"],"success":e["success"],"adverse_outcome":not e["success"],
              "steps":e["steps"],"termination_reason":e["termination_reason"],
              "chunk_boundaries":len(ids),"gp_mean_error_rmse":float(np.sqrt(np.mean((e["Y"]-e["mean"])**2))),
              "signals":{}}
        for name,values in signal_map.items():
            peak=float(values[ids].max());first=np.flatnonzero(values[ids]>=threshold_map[name])
            item["signals"][name]={"mean":float(values[ids].mean()),"max":peak,
                "threshold":float(threshold_map[name]),"request_count":int(np.sum(values[ids]>=threshold_map[name])),
                "request_episode":bool(len(first)),"first_request_step":int(e["chunk_steps"][first[0]]) if len(first) else None,
                "lead_steps_to_episode_end":int(e["steps"]-e["chunk_steps"][first[0]]) if len(first) else None}
        episode_rows.append(item)

    y_episode=np.asarray([int(not e["success"]) for e in episodes])
    episode_scores={name:np.asarray([item["signals"][name]["max"] for item in episode_rows])
                    for name in signal_map}
    risk={name:safe_rank_metrics(y_episode,value) for name,value in episode_scores.items()}
    episode_index_per_row=np.concatenate([np.full(len(e["X"]),i,dtype=int) for i,e in enumerate(episodes)])
    random_matched={name:matched_random_budget(y_episode,episode_index_per_row,
        int(sum(item["signals"][name]["request_count"] for item in episode_rows)),
        seed=20261104+index) for index,name in enumerate(signal_map)}
    totals={}
    for name in signal_map:
        flags=np.asarray([item["signals"][name]["request_episode"] for item in episode_rows],dtype=bool)
        tp=int(np.sum(flags & (y_episode==1)));fp=int(np.sum(flags & (y_episode==0)))
        fn=int(np.sum(~flags & (y_episode==1)));tn=int(np.sum(~flags & (y_episode==0)))
        totals[name]={"request_boundaries":int(sum(item["signals"][name]["request_count"] for item in episode_rows)),
                      "episode_confusion_vs_final_outcome":{"tp":tp,"fp":fp,"fn":fn,"tn":tn},
                      "note":"A request in an eventually successful episode is not proven unnecessary; no counterfactual assistance was run."}

    correlations={name:cluster_bootstrap_spearman(values,err,ep_keys,seed=4901+i)
                  for i,(name,values) in enumerate((("uncertainty",U),("state_novelty",H)))}
    standardized_residual=(Y-M)/S
    by_task={}
    for task_id in sorted(np.unique(task_ids)):
        mask=task_ids==task_id
        by_task[str(int(task_id))]={
            "chunk_start_pairs":int(mask.sum()),
            "gp_mae":float(np.mean(np.abs(Y[mask]-M[mask]))),
            "gp_rmse":float(np.sqrt(np.mean((Y[mask]-M[mask])**2))),
            "ridge_mae":float(np.mean(np.abs(Y[mask]-ridge_mean[mask]))),
            "ridge_rmse":float(np.sqrt(np.mean((Y[mask]-ridge_mean[mask])**2))),
            "coverage90":float(np.mean(np.abs(Y[mask]-M[mask])<=Z90*S[mask])),
            "coverage95":float(np.mean(np.abs(Y[mask]-M[mask])<=Z95*S[mask])),
        }
    report={
      "experiment_id":"gp_uncertainty_selective_assistance_heldout_2026-11-04",
      "heldout_episode_count":len(episodes),"successes":int(sum(e["success"] for e in episodes)),
      "adverse_episodes":int(sum(not e["success"] for e in episodes)),
      "outcome_definition":"LIBERO episode is_success false, including the 280-step environment time limit.",
      "action_surrogate_fidelity":{
        "chunk_start_pairs":int(len(Y)),"gp_overall_mae":float(np.mean(np.abs(Y-M))),
        "gp_overall_rmse":float(np.sqrt(np.mean((Y-M)**2))),
        "ridge_overall_mae":float(np.mean(np.abs(Y-ridge_mean))),
        "ridge_overall_rmse":float(np.sqrt(np.mean((Y-ridge_mean)**2))),
        "mae_by_action_dim":gp_mae.tolist(),"rmse_by_action_dim":gp_rmse.tolist(),
        "ridge_mae_by_action_dim":ridge_mae.tolist(),"ridge_rmse_by_action_dim":ridge_rmse.tolist(),
        "disagreement_equals_standardized_surrogate_error":bool(np.allclose(D,err)),
        "disagreement_note":"By construction it is the discrepancy between GP surrogate and the observed pi0.5 horizon-zero target; it is not VLA task error."
      },
      "per_task_action_fidelity_and_coverage":by_task,
      "standardized_residual_distribution":{
        "definition":"(VLA horizon-zero target minus GP mean) / GP predictive standard deviation, per action dimension.",
        "mean_by_action_dim":np.mean(standardized_residual,axis=0).tolist(),
        "std_by_action_dim":np.std(standardized_residual,axis=0).tolist(),
        "absolute_q90_by_action_dim":np.quantile(np.abs(standardized_residual),.90,axis=0).tolist(),
      },
      "uncertainty_calibration":{
        "predictive_variance":"latent posterior variance plus fixed observation noise, in raw checkpoint-normalized action units.",
        "coverage90_by_action_dim":cov90.tolist(),"coverage95_by_action_dim":cov95.tolist(),
        "overall_coverage90":float(np.mean(np.abs(Y-M)<=Z90*S)),
        "overall_coverage95":float(np.mean(np.abs(Y-M)<=Z95*S)),
        "validation_multiplier":multiplier,
        "calibrated_coverage90_by_action_dim":cal_cov90.tolist(),
        "calibrated_coverage95_by_action_dim":cal_cov95.tolist(),
        "overall_calibrated_coverage90":float(np.mean(np.abs(Y-M)<=Z90*S*multiplier)),
        "overall_calibrated_coverage95":float(np.mean(np.abs(Y-M)<=Z95*S*multiplier)),
        "interpretation":"Empirical held-out coverage only; the validation scaling is not a conformal guarantee."
      },
      "error_signal_relationships":{
        "uncertainty_vs_action_error_cluster_bootstrap_spearman":correlations["uncertainty"],
        "state_novelty_vs_action_error_cluster_bootstrap_spearman":correlations["state_novelty"],
        "disagreement_vs_action_error_spearman":1.0,
        "note":"Chunk boundaries are clustered by episode; cluster bootstrap resamples whole episodes. Only 10 held-out episodes."
      },
      "task_risk_discrimination":{
        "labels_by_episode":[{"episode_key":e["episode_key"],"task_id":e["task_id"],"success":e["success"],"steps":e["steps"]} for e in episodes],
        "positive_count":int(y_episode.sum()),"negative_count":int((y_episode==0).sum()),
        "episode_score":"Maximum chunk-boundary score over the full episode, fixed for all signal families.",
        "ranking_metrics":risk,"threshold_metrics":totals,
        "random_rank_reference":{"expected_auroc":0.5,"average_precision_prevalence":float(np.mean(y_episode))},
        "matched_random_budget_comparison":random_matched,
        "interpretation":"Only one timeout occurred; all ranking estimates are exploratory and cannot validate a general risk detector."
      },
      "episodes":episode_rows,
      "resource":{"gpu_name":"RTX 5070 Ti Laptop 12GB","peak_torch_allocated_bytes":7125786624,
                  "peak_torch_reserved_bytes":7291797504,"renderer":"OSMesa"}
    }
    (output/"report.json").write_text(json.dumps(report,indent=2)+"\n")
    (output/"episodes.csv").write_text("episode_key,task_id,state_index,seed,success,steps,termination_reason,chunk_boundaries,gp_mean_error_rmse\n"+
      "".join(f"{e['episode_key']},{e['task_id']},{e['state_index']},{e['seed']},{int(e['success'])},{e['steps']},{e['termination_reason']},{e['chunk_boundaries']},{e['gp_mean_error_rmse']:.8f}\n" for e in episode_rows))
    np.savez_compressed(output/"chunk_signals.npz",X=X,Y_vla=Y,gp_mean=M,gp_std=S,
      uncertainty=U,disagreement=D,state_novelty=H,error=err,episode_key=ep_keys,task_id=task_ids,adverse=labels)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,values) in zip(axes,signal_map.items()):
        for task_id,marker,label in ((1,"o","task 1"),(4,"s","task 4")):
            mask=task_ids==task_id
            ax.scatter(values[mask],err[mask],marker=marker,label=label,alpha=.65)
        ax.set(xlabel=name.replace("_"," "),ylabel="standardized GP action error",title=f"{name} vs action error")
        ax.grid(alpha=.2);ax.legend()
    fig.tight_layout();fig.savefig(output/"heldout_signal_vs_error.png",dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for ax,(name,values) in zip(axes,episode_scores.items()):
        colors=["tab:red" if y else "tab:blue" for y in y_episode]
        ax.scatter(values,np.arange(len(values)),c=colors)
        for i,e in enumerate(episodes):ax.annotate(f"T{e['task_id']} S{e['state_index']}",(values[i],i),xytext=(3,0),textcoords="offset points",fontsize=7)
        ax.axvline(threshold_map[name],color="k",linestyle="--",label="validation p90")
        ax.set(xlabel=f"episode max {name}",ylabel="episode index",title="red=timeout, blue=success")
        ax.grid(alpha=.2)
    fig.tight_layout();fig.savefig(output/"heldout_episode_scores.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,4))
    ax.hist(standardized_residual.ravel(),bins=35,density=True,alpha=.75,label="held-out standardized residuals")
    grid=np.linspace(-4,4,300)
    ax.plot(grid,np.exp(-grid**2/2)/np.sqrt(2*np.pi),"k--",label="standard normal reference")
    ax.set(xlabel="standardized residual",ylabel="density",title="Held-out residual distribution")
    ax.grid(alpha=.2)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output/"standardized_residual_distribution.png",dpi=160)
    plt.close(fig)
    with (output/"per_action_metrics.csv").open("w",newline="") as f:
        w=csv.writer(f);w.writerow(["action_dim","gp_mae","gp_rmse","ridge_mae","ridge_rmse","coverage90","coverage95","calibrated_coverage90","calibrated_coverage95"])
        for d in range(7):w.writerow([d,gp_mae[d],gp_rmse[d],ridge_mae[d],ridge_rmse[d],cov90[d],cov95[d],cal_cov90[d],cal_cov95[d]])
    print(json.dumps({k:report[k] for k in ("heldout_episode_count","successes","adverse_episodes","action_surrogate_fidelity","uncertainty_calibration","error_signal_relationships","task_risk_discrimination")},indent=2))


if __name__=="__main__":main()
