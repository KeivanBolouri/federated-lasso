"""Rebuild the article's tables, figures and numbers from committed result CSVs.

Run from any working directory. This script never reruns or overwrites raw
simulations. The independent simulation driver is ``run_study.py``.
"""
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CODE = Path(__file__).resolve().parent
MS = str(CODE.parent) + "/"
E_GRID = [1, 2, 3, 5, 8, 10, 15, 20]
SC = ["Independent", "Correlated", "Heterogeneous"]
FEDM = ["FedAvg", "FedAvg-1SE", "FedAvg-ST", "FedAvg-ST-min", "FedAvg-P"]
BASE = ["Pooled", "Pooled-1SE", "Local-only", "Local-only-1SE", "One-shot",
        "One-shot-1SE", "ADMM", "ADMM-CV", "ADMM-CV-1SE"]
MET = ["active", "TP", "FP", "precision", "recall", "F1", "jaccard",
       "coef_err", "test_mse", "rounds", "local_epochs", "vector_kib",
       "scalar_kib", "upload_kib", "selected_rounds", "selected_local_epochs",
       "selected_vector_kib", "tuning_rounds", "total_rounds"]
SEM = ["F1", "jaccard", "coef_err", "test_mse", "recall", "precision", "active"]
C = {"FedAvg": "#1b4965", "FedAvg-1SE": "#00867d", "FedAvg-ST": "#b34a57",
     "FedAvg-P": "#ca8617", "ADMM": "#575a9c", "Pooled-1SE": "#444444",
     "Local-only-1SE": "#a174ad"}


def load(path=CODE / "sim_all_raw.csv"):
    d = pd.read_csv(path)
    d["scenario"] = d["scenario"].replace({"IID": "Independent"})
    if d.duplicated(["scenario", "rep", "method", "E"]).any():
        raise ValueError("Duplicate simulation keys; refusing to pool duplicate runs")
    if d["precision"].isna().any():
        raise ValueError("Undefined precision: regenerate with the documented zero convention")
    return d


def agg(d):
    g = d.groupby(["scenario", "method", "E"], dropna=False)
    return (g.size().to_frame("n")
            .join(g[[m for m in MET if m in d]].mean().add_suffix("_m"))
            .join(g[SEM].sem().add_suffix("_se")).reset_index())


def f(x, k=3):
    return "---" if pd.isna(x) else f"{x:.{k}f}"


def pick(a, s, method, E=None):
    r = a[(a.scenario == s) & (a.method == method) &
          (a.E.isna() if E is None else (a.E == E))]
    return None if len(r) == 0 else r.iloc[0]


def write_table(name, lines):
    (CODE.parent / "tables" / name).write_text("\n".join(lines) + "\n")


def long_head(columns, caption, label, headings):
    return [r"\begingroup\footnotesize\setlength{\tabcolsep}{4pt}",
            r"\begin{longtable}{" + columns + "}",
            r"\caption{" + caption + r"}\label{" + label + r"}\\",
            r"\toprule", headings + r" \\", r"\midrule", r"\endfirsthead",
            r"\toprule", headings + r" \\", r"\midrule", r"\endhead",
            r"\bottomrule", r"\endfoot"]


def main_table(a):
    cap = (r"Simulation results, 200 replicates per design ($p=600$, $s=30$). "
           r"$F_1$ is relative to the true support; parentheses give Monte Carlo standard errors "
           r"for $F_1$ and test MSE. "
           r"Precision is zero for empty selected models. The -1SE suffix denotes one-standard-error "
           r"penalty selection; ST and P use one-SE threshold selection after local minimum-MSE "
           r"penalty selection. ADMM uses the inherited fixed penalty $\bar\lambda$. "
           r"Cost accounting is reported separately in Supplementary Table~S1.")
    L = long_head("llrrrrr", cap, "tab:main",
                  r"Design & Method & Active & Precision & Recall & $F_1$ (MCSE) & Test MSE (MCSE)")
    L[0] = L[0].replace(r"{4pt}", r"{3pt}")
    L[0] += r"\renewcommand{\arraystretch}{0.93}"
    order = [(m, None) for m in BASE] + [(m, e) for m in
              ["FedAvg", "FedAvg-1SE", "FedAvg-ST", "FedAvg-P"] for e in (1, 20)]
    for si, s in enumerate(SC):
        for i, (m, e) in enumerate(order):
            r = pick(a, s, m, e)
            if r is None:
                continue
            nm = m if e is None else f"{m} ($E={e}$)"
            L.append(f"{s if i == 0 else ''} & {nm} & {r.active_m:.1f} & "
                     f"{r.precision_m:.3f} & {r.recall_m:.3f} & "
                     f"{r.F1_m:.3f} ({r.F1_se:.3f}) & {r.test_mse_m:.2f} ({r.test_mse_se:.2f}) \\\\")
        if si < 2:
            L.append(r"\midrule")
    L += [r"\end{longtable}\endgroup"]
    write_table("main_results.tex", L)


def full_table(a):
    cap = (r"Full simulation grid. Means over 200 replicates; Monte Carlo standard errors for "
           r"$F_1$ in parentheses. Fit rounds and CD epochs include all five candidate fits for "
           r"FedAvg-P. Upload includes their coefficient vectors and the validation scalars "
           r"(2.81 KiB for ST; 0.35 KiB for P). Local penalty-search computation and all "
           r"downloads are excluded. ADMM-CV is computed centrally using the equivalent pooled "
           r"objective, so no federated cost is assigned. Other metrics and selected-fit costs "
           r"are included in the accompanying CSV files.")
    L = long_head("llrrrrrrr", cap, "tab:full",
                  r"Design & Method & $E$ & Active & $F_1$ (MCSE) & $\|\hat\beta-\beta^\star\|_2$ & Fit rounds & CD epochs & KiB")
    order = [(m, None) for m in BASE] + [(m, e) for m in FEDM for e in E_GRID]
    for si, s in enumerate(SC):
        for i, (m, e) in enumerate(order):
            r = pick(a, s, m, e)
            if r is None:
                continue
            L.append(f"{s if i == 0 else ''} & {m} & {f(e,0)} & {r.active_m:.1f} & "
                     f"{r.F1_m:.3f} ({r.F1_se:.3f}) & {r.coef_err_m:.2f} & "
                     f"{f(r.rounds_m,1)} & {f(r.local_epochs_m,1)} & {f(r.upload_kib_m,1)} \\\\")
        if si < 2:
            L.append(r"\midrule")
    L += [r"\end{longtable}\endgroup"]
    write_table("full_results.tex", L)


def paired_table(d):
    rows = []
    for s in SC:
        for m in ["FedAvg", "FedAvg-1SE", "FedAvg-ST", "FedAvg-P"]:
            p = d[(d.scenario == s) & (d.method == m)].pivot(index="rep", columns="E", values="F1")
            if 1 not in p or 20 not in p:
                continue
            delta = p[20] - p[1]
            rows.append(dict(scenario=s, method=m, n=len(delta), difference=delta.mean(),
                             mcse=delta.std(ddof=1) / np.sqrt(len(delta))))
    p = pd.DataFrame(rows)
    p.to_csv(CODE / "paired_epoch_results.csv", index=False)
    L = [r"\begin{table}[tb]\centering\small",
         r"\caption{Paired change in true-support $F_1$, $E=20$ minus $E=1$. Each difference is computed within replicate before averaging; MCSE is the standard deviation of the paired differences divided by $\sqrt{200}$.}",
         r"\label{tab:paired}\begin{tabular}{llrr}\toprule",
         r"Design & Method & Mean change & Paired MCSE \\\midrule"]
    for s in SC:
        for i, r in enumerate(p[p.scenario == s].itertuples()):
            L.append(f"{s if i == 0 else ''} & {r.method} & {r.difference:+.4f} & {r.mcse:.4f} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    write_table("paired_epochs.tex", L)
    return p


def real_table():
    d = pd.read_csv(CODE / "real_data_results.csv")
    d = d[d.E.isna() | d.E.isin([1, 2, 3, 20])]
    cap = (r"Fixed synthetic benchmark ($n=800$, $p=600$, $K=3$). The original generator "
           r"and true coefficients were not supplied. Agreement is with the pooled minimum-MSE "
           r"Lasso support (\Srefactive\ variables), not with ground truth. $F$ is the common "
           r"evaluation objective. Upload includes validation scalars and all five candidate "
           r"fits for P; ADMM-CV costs are not measured.")
    L = long_head("lrrrrrr", cap, "tab:real",
                  r"Method & $E$ & Active & Precision & Recall & $F_1$ & $F$")
    for r in d.itertuples():
        L.append(f"{r.method} & {f(r.E,0)} & {r.active} & {r.precision:.3f} & "
                 f"{r.recall:.3f} & {r.F1:.3f} & {r.obj:.3f} \\\\")
    L += [r"\end{longtable}\endgroup"]
    # Costs are retained in the CSV rather than crowded into this table.
    L[2] = L[2].replace("Upload includes validation scalars and all five candidate fits for P; ADMM-CV costs are not measured.",
                         r"Selected epoch counts are shown; the complete grid and costs, including all five P candidates, are supplied in \texttt{code/real\_data\_results.csv}.")
    write_table("real_data.tex", L)


def eps_table():
    d = pd.read_csv(CODE / "eps_sensitivity.csv")
    L = [r"\begin{table}[htbp]\centering\small",
         r"\caption{Active coefficients in the fixed synthetic benchmark at different activity thresholds. All methods may retain exact zeros; increasing the threshold also removes nonzero coefficients, including for ST and ADMM.}",
         r"\label{tab:eps}\begin{tabular}{lrrrrr}\toprule",
         r"Method & $E$ & $\epsilon=10^{-6}$ & $10^{-4}$ & $10^{-2}$ & $10^{-1}$ \\\midrule"]
    for _, r in d.iterrows():
        L.append(f"{r.method} & {f(r.E,0)} & {int(r['eps_1e-06'])} & "
                 f"{int(r['eps_0.0001'])} & {int(r['eps_0.01'])} & {int(r['eps_0.1'])} \\\\")
    L += [r"\bottomrule\end{tabular}\end{table}"]
    write_table("eps_sensitivity.tex", L)


def save_figure(fig, name):
    for extension in ("pdf", "png"):
        target = CODE.parent / "figures" / f"{name}.{extension}"
        temporary = target.with_suffix(f".{extension}.tmp")
        fig.savefig(temporary, format=extension, dpi=300, bbox_inches="tight")
        temporary.replace(target)
    plt.close(fig)


def figures(a):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.18,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(2, 3, figsize=(9.4, 5.4), sharex=True)
    for c, s in enumerate(SC):
        for m in ["FedAvg", "FedAvg-1SE", "FedAvg-ST", "FedAvg-P"]:
            r = a[(a.scenario == s) & (a.method == m)].sort_values("E")
            if r.empty:
                continue
            ax[0,c].errorbar(r.E, r.F1_m, yerr=r.F1_se, marker="o", ms=3, lw=1.2,
                            color=C[m], label=m, capsize=2)
            ax[1,c].plot(r.E, r.active_m, marker="o", ms=3, lw=1.2, color=C[m])
        for m, ls in [("ADMM", "--"), ("Pooled-1SE", ":"), ("Local-only-1SE", "-.")]:
            r = pick(a,s,m)
            ax[0,c].axhline(r.F1_m, ls=ls, lw=1.0, color=C[m], label=m)
            ax[1,c].axhline(r.active_m, ls=ls, lw=1.0, color=C[m])
        ax[0,c].set_title(s)
        ax[1,c].set_xlabel("Local CD epochs $E$")
        ax[1,c].set_xscale("log"); ax[1,c].set_xticks(E_GRID, labels=E_GRID)
    ax[0,0].set_ylabel("$F_1$ against true support")
    ax[1,0].set_ylabel("Active coefficients")
    h, lab = ax[0,0].get_legend_handles_labels()
    fig.legend(h,lab,loc="lower center",ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.11,1,1)); save_figure(fig,"fig1_support")

    fig, ax = plt.subplots(1,3,figsize=(9.4,2.9))
    for c,s in enumerate(SC):
        r = a[(a.scenario==s)&(a.method=="FedAvg")].sort_values("E")
        ax[c].plot(r.E,r.vector_kib_m,"o-",color=C["FedAvg"],ms=3)
        a2=ax[c].twinx(); a2.grid(False)
        a2.plot(r.E,r.local_epochs_m,"s--",color=C["FedAvg-ST"],ms=3)
        ax[c].set_title(s); ax[c].set_xscale("log")
        ax[c].set_xticks(E_GRID,labels=E_GRID); ax[c].set_xlabel("Local CD epochs $E$")
        if c==0: ax[c].set_ylabel("Upload (KiB), solid")
        if c==2: a2.set_ylabel("CD epochs per site, dashed")
    fig.tight_layout(); save_figure(fig,"fig2_cost")

    d=pd.read_csv(CODE/"real_data_results.csv")
    fig, ax=plt.subplots(1,3,figsize=(9.4,2.8))
    for m in ["FedAvg","FedAvg-ST","FedAvg-P"]:
        r=d[d.method==m].sort_values("E")
        for j,metric in enumerate(["active","F1","obj"]):
            ax[j].plot(r.E,r[metric],"o-",ms=3,color=C[m],label=m)
    adm=d[d.method=="ADMM"].iloc[0]
    for j,metric in enumerate(["active","F1","obj"]):
        ax[j].axhline(adm[metric],ls="--",color=C["ADMM"],lw=1,label="ADMM")
        ax[j].set_xscale("log"); ax[j].set_xticks(E_GRID,labels=E_GRID)
        ax[j].set_xlabel("Local CD epochs $E$")
    for j,label in enumerate(["Active coefficients","$F_1$ against pooled support","Evaluation objective $F$"]):
        ax[j].set_ylabel(label)
    fig.legend(*ax[0].get_legend_handles_labels(),loc="lower center",ncol=4,frameon=False,fontsize=8)
    fig.tight_layout(rect=(0,.13,1,1)); save_figure(fig,"fig3_realdata")


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw",type=Path,default=CODE/"sim_all_raw.csv")
    args=parser.parse_args()
    for name in ["tables","figures"]: (CODE.parent/name).mkdir(exist_ok=True)
    d=load(args.raw); a=agg(d)
    a.to_csv(CODE/"sim_summary_new.csv",index=False)
    main_table(a); full_table(a); paired_table(d); real_table(); eps_table(); figures(a); macros(a)
    print(f"Rebuilt outputs from {len(d):,} raw rows; {len(a)} summary rows.")


def macros(a):
    d = pd.read_csv(CODE / "real_data_results.csv").set_index(["method", "E"])
    g = lambda m, E=np.nan: d.loc[(m, E)] if (m, E) in d.index else d.xs(m, level=0).iloc[0]
    fa1, fa20, st1 = g("FedAvg", 1.0), g("FedAvg", 20.0), g("FedAvg-ST", 1.0)
    fp1 = g("FedAvg-P", 1.0)
    adm, loc, pool = g("ADMM"), g("Local-only"), g("Pooled")
    pool1, loc1 = g("Pooled-1SE"), g("Local-only-1SE")
    acv1 = g("ADMM-CV-1SE")
    m = dict(Slamone="0.2046", Slamtwo="0.3635", Slamthree="0.1953",
             Slambar="0.2504", Slampooled="0.1778",
             Srefactive=f"{int(pool.active)}",
             Sfedactive=f"{int(fa1.active)}", SfedFone=f"{fa1.F1:.3f}",
             SlocalFone=f"{loc.F1:.3f}", SlocalseFone=f"{loc1.F1:.3f}",
             SpoolseFone=f"{pool1.F1:.3f}", SpoolseAct=f"{int(pool1.active)}",
             Sstactive=f"{int(st1.active)}", SstFone=f"{st1.F1:.3f}",
             SpFone=f"{fp1.F1:.3f}", Spactive=f"{int(fp1.active)}",
             Sadmmactive=f"{int(adm.active)}", Sadmmtp=f"{int(adm.TP)}",
             Sadmmfp=f"{int(adm.FP)}", SadmmFone=f"{adm.F1:.3f}",
             SadmmcvFone=f"{acv1.F1:.3f}", SadmmcvAct=f"{int(acv1.active)}",
             Sadmmobj=f"{adm.obj:.3f}", Sfedobj=f"{fa1.obj:.3f}",
             Sfedobjtwenty=f"{fa20.obj:.3f}",
             Sfedobjtwo=f"{g('FedAvg', 2.0).obj:.3f}",
             Sfedobjrange=f"{d.xs('FedAvg',level=0).obj.max()-d.xs('FedAvg',level=0).obj.min():.3f}",
             Sgapabs=f"{fa1.obj-adm.obj:.3f}",
             Sgappct=f"{100*(fa1.obj-adm.obj)/adm.obj:.1f}\\%",
             Sroundsone=f"{int(fa1.rounds)}", Sroundstwenty=f"{int(fa20.rounds)}",
             Skibone=f"{fa1.vector_kib:.1f}", Skibtwenty=f"{fa20.vector_kib:.1f}",
             Sepochsone=f"{int(fa1.epochs)}", Sepochstwenty=f"{int(fa20.epochs)}",
             Sstscalar=f"{float(d.xs('FedAvg-ST',level=0).scalar_kib.iloc[0]):.1f}",
             SstFtwenty=f"{g('FedAvg-ST', 20.0).F1:.3f}")
    for s, tag in zip(SC, ["Ind", "Cor", "Het"]):
        p = lambda mth, E=None: pick(a, s, mth, E)
        pairs = {"Fed": ("FedAvg", 1), "FedT": ("FedAvg", 20),
                 "FedSe": ("FedAvg-1SE", 1), "FedSeT": ("FedAvg-1SE", 20), "OneSe": ("One-shot-1SE", None),
                 "St": ("FedAvg-ST", 1), "StT": ("FedAvg-ST", 20),
                 "Prox": ("FedAvg-P", 1), "ProxT": ("FedAvg-P", 20),
                 "Pool": ("Pooled", None), "PoolSe": ("Pooled-1SE", None),
                 "Loc": ("Local-only", None), "LocSe": ("Local-only-1SE", None),
                 "Admm": ("ADMM", None), "AdmmCv": ("ADMM-CV", None),
                 "AdmmCvSe": ("ADMM-CV-1SE", None), "One": ("One-shot", None)}
        for key, (mth, E) in pairs.items():
            r = p(mth, E)
            if r is None:
                continue
            m[f"M{tag}{key}F"] = f"{r.F1_m:.3f}"
            m[f"M{tag}{key}Act"] = f"{r.active_m:.0f}"
            m[f"M{tag}{key}Se"] = f"{r.F1_se:.3f}"
            m[f"M{tag}{key}Prec"] = f"{r.precision_m:.3f}"
            m[f"M{tag}{key}Rec"] = f"{r.recall_m:.3f}"
            m[f"M{tag}{key}Err"] = f"{r.coef_err_m:.2f}"
            m[f"M{tag}{key}Mse"] = f"{r.test_mse_m:.1f}"
        for e, suffix in [(1, "One"), (20, "Twenty")]:
            r = p("FedAvg-P", e)
            m[f"M{tag}ProxAllRounds{suffix}"] = f"{r.rounds_m:.1f}"
            m[f"M{tag}ProxSelectedRounds{suffix}"] = f"{r.selected_rounds_m:.1f}"
            m[f"M{tag}ProxUpload{suffix}"] = f"{r.upload_kib_m:.1f}"
        m[f"M{tag}NRep"] = f"{int(p('FedAvg', 1).n)}"
        m[f"M{tag}RoundsOne"] = f"{p('FedAvg',1).rounds_m:.1f}"
        m[f"M{tag}RoundsTwenty"] = f"{p('FedAvg',20).rounds_m:.1f}"
        m[f"M{tag}EpochsOne"] = f"{p('FedAvg',1).local_epochs_m:.1f}"
        m[f"M{tag}EpochsTwenty"] = f"{p('FedAvg',20).local_epochs_m:.1f}"
        m[f"M{tag}AdmmRounds"] = f"{p('ADMM').rounds_m:.0f}"
        m[f"M{tag}ActDrop"] = f"{100*(1-p('FedAvg',20).active_m/p('FedAvg',1).active_m):.1f}\\%"
        m[f"M{tag}StScalar"] = f"{p('FedAvg-ST',1).scalar_kib_m:.1f}"
        m[f"M{tag}FedVectorOne"] = f"{p('FedAvg',1).vector_kib_m/p('FedAvg',1).rounds_m:.1f}"
    meta_path = CODE / "real_data_metadata.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for key, val in zip(["Slamone", "Slamtwo", "Slamthree"], meta["local_penalties"]):
            m[key] = f"{val:.4f}"
        m["Slambar"] = f"{meta['lambda_bar']:.4f}"
        m["Slampooled"] = f"{meta['pooled_penalty']:.4f}"
    with open(MS + "numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))


if __name__ == "__main__":
    main()
