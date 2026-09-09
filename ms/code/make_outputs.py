"""Build LaTeX tables, figures and \newcommand macros for the manuscript."""
import glob, re, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = "/home/claude/work/ms/"
E_GRID = [1, 2, 3, 5, 8, 10, 15, 20]
SC = ["Independent", "Correlated", "Heterogeneous"]
FEDM = ["FedAvg", "FedAvg-ST", "FedAvg-ST-min", "FedAvg-P"]
BASE = ["Pooled", "Pooled-1SE", "Local-only", "Local-only-1SE", "One-shot",
        "ADMM", "ADMM-CV", "ADMM-CV-1SE"]
MET = ["active", "TP", "FP", "precision", "recall", "F1", "jaccard",
       "coef_err", "test_mse", "rounds", "local_epochs", "vector_kib",
       "scalar_kib"]
SEM = ["F1", "jaccard", "coef_err", "test_mse", "recall", "precision", "active"]


def load():
    fs = [f for f in sorted(glob.glob("out_*.csv")) if "pilot" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["scenario"] = d["scenario"].replace({"IID": "Independent"})
    return d


def agg(d):
    g = d.groupby(["scenario", "method", "E"], dropna=False)
    return (g.size().to_frame("n")
            .join(g[MET].mean().add_suffix("_m"))
            .join(g[SEM].sem().add_suffix("_se")).reset_index())


def f(x, k=3):
    return "---" if pd.isna(x) else f"{x:.{k}f}"


def pick(a, s, mth, E=None):
    r = a[(a.scenario == s) & (a.method == mth) &
          (a.E.isna() if E is None else (a.E == E))]
    return None if not len(r) else r.iloc[0]


def main_table(a):
    L = [r"\begin{table}[p]", r"\centering", r"\footnotesize",
         r"\caption{Monte Carlo results over 200 replicates per design "
         r"($p=600$, $s=30$, $K=3$, $m=(400,240,160)$). Support metrics are "
         r"against the true support at $\epsilon=10^{-4}$; Monte Carlo "
         r"standard errors for $F_1$ in parentheses. Rows marked "
         r"\textsc{-1SE} select their penalty by the one-standard-error rule, "
         r"the rule the federated thresholds use; the unmarked rows minimise "
         r"validation MSE. Federated schedules are shown at $E=1$ and $E=20$; "
         r"the full grid is in Table~\ref{tab:full}.}",
         r"\label{tab:main}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Design & Method & Active & Prec. & Recall & $F_1$ & "
         r"$\|\hat\beta-\beta^\star\|_2$ & Test MSE & Rounds \\", r"\midrule"]
    order = [(m, None) for m in BASE] + \
            [(m, E) for m in ["FedAvg", "FedAvg-ST", "FedAvg-P"] for E in (1, 20)]
    for si, s in enumerate(SC):
        for i, (mth, E) in enumerate(order):
            r = pick(a, s, mth, E)
            if r is None:
                continue
            nm = mth if E is None else f"{mth} ($E={E}$)"
            L.append(f"{s if i==0 else ''} & {nm} & {r.active_m:.1f} & "
                     f"{f(r.precision_m)} & {f(r.recall_m)} & "
                     f"{f(r.F1_m)} ({r.F1_se:.3f}) & {f(r.coef_err_m,2)} & "
                     f"{f(r.test_mse_m,2)} & {f(r.rounds_m,1)} \\\\")
            if i == len(BASE) - 1:
                L.append(r"\addlinespace")
        L.append(r"\midrule" if si < len(SC) - 1 else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(MS + "tables/main_results.tex", "w").write("\n".join(L))


def full_table(a):
    L = [r"\footnotesize", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{longtable}{llrrrrrrrrr}",
         r"\caption{Complete Monte Carlo results, 200 replicates per design. "
         r"MCSEs for $F_1$ in parentheses. Upload is KiB of uploaded "
         r"coefficient vectors; epochs are total coordinate-descent passes "
         r"per site. The threshold-selection step adds 2.81~KiB of scalars "
         r"for \textsc{FedAvg-ST} and 0.35~KiB for \textsc{FedAvg-P}, and "
         r"nothing for the other methods.}"
         r"\label{tab:full}\\", r"\toprule",
         r"Design & Method & $E$ & Active & Prec. & Recall & $F_1$ & Jacc. "
         r"& Rounds & Epochs & Upload \\", r"\midrule",
         r"\endfirsthead", r"\toprule",
         r"Design & Method & $E$ & Active & Prec. & Recall & $F_1$ & Jacc. "
         r"& Rounds & Epochs & Upload \\", r"\midrule",
         r"\endhead", r"\bottomrule", r"\endfoot"]
    for si, s in enumerate(SC):
        first = True
        for mth in BASE:
            r = pick(a, s, mth)
            if r is None:
                continue
            L.append(f"{s if first else ''} & {mth} & --- & {r.active_m:.1f} & "
                     f"{f(r.precision_m)} & {f(r.recall_m)} & {f(r.F1_m)} & "
                     f"{f(r.jaccard_m)} & {f(r.rounds_m,1)} & --- & "
                     f"{f(r.vector_kib_m,1)} \\\\")
            first = False
        for mth in FEDM:
            for E in E_GRID:
                r = pick(a, s, mth, E)
                if r is None:
                    continue
                L.append(f" & {mth} & {E} & {r.active_m:.1f} & {f(r.precision_m)} & "
                         f"{f(r.recall_m)} & {f(r.F1_m)} ({r.F1_se:.3f}) & "
                         f"{f(r.jaccard_m)} & {f(r.rounds_m,1)} & "
                         f"{f(r.local_epochs_m,1)} & {f(r.vector_kib_m,1)} \\\\")
        L.append(r"\midrule" if si < len(SC) - 1 else "")
    L += [r"\end{longtable}"]
    open(MS + "tables/full_results.tex", "w").write("\n".join(L))


def real_table():
    d = pd.read_csv("real_data_results.csv")
    L = [r"\begin{table}[p]", r"\centering", r"\footnotesize",
         r"\caption{Three-site data illustration ($n=800$ training, $p=600$, "
         r"$K=3$). Agreement is with the pooled Lasso reference support "
         r"(\Srefactive\ active predictors), not with a known truth. $F$ is "
         r"the aggregated objective \eqref{eq:global}; consensus ADMM "
         r"minimises it exactly. Upload is coefficient-vector traffic; the "
         r"threshold step adds \Sstscalar~KiB of scalars.}",
         r"\label{tab:real}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Method & $E$ & Active & TP & FP & Prec. & Recall & $F_1$ & $F$ "
         r"& Vec.\ KiB \\", r"\midrule"]
    for _, r in d.iterrows():
        E = "---" if pd.isna(r.E) else str(int(r.E))
        L.append(f"{r.method} & {E} & {int(r.active)} & {int(r.TP)} & "
                 f"{int(r.FP)} & {f(r.precision)} & {f(r.recall)} & "
                 f"{f(r.F1)} & {f(r.obj,3)} & {f(r.vector_kib,1)} \\\\")
        if r.method == "ADMM-CV-1SE":
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(MS + "tables/real_data.tex", "w").write("\n".join(L))


def eps_table():
    d = pd.read_csv("eps_sensitivity.csv")
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Number of coefficients declared active in the data "
         r"illustration as the activity threshold $\epsilon$ varies. The "
         r"plain average has no exact zeros, so its apparent sparsity is a "
         r"choice of $\epsilon$; the thresholded and ADMM solutions do.}",
         r"\label{tab:eps}", r"\begin{tabular}{lrrrrr}", r"\toprule",
         r"Method & $E$ & $\epsilon=10^{-6}$ & $10^{-4}$ & $10^{-2}$ & "
         r"$10^{-1}$ \\", r"\midrule"]
    for _, r in d.iterrows():
        E = "---" if pd.isna(r.E) else str(int(r.E))
        L.append(f"{r.method} & {E} & {int(r['eps_1e-06'])} & "
                 f"{int(r['eps_0.0001'])} & {int(r['eps_0.01'])} & "
                 f"{int(r['eps_0.1'])} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(MS + "tables/eps_sensitivity.tex", "w").write("\n".join(L))


C = {"FedAvg": "#1b4965", "FedAvg-ST": "#c1666b", "FedAvg-P": "#e09f3e",
     "ADMM": "#4c956c", "Pooled-1SE": "#7a7a7a", "Local-only-1SE": "#8a6fa8"}


def figures(a):
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "figure.dpi": 160})
    fig, ax = plt.subplots(2, 3, figsize=(9.4, 5.2), sharex=True)
    for c, s in enumerate(SC):
        for mth in ["FedAvg", "FedAvg-ST", "FedAvg-P"]:
            r = a[(a.scenario == s) & (a.method == mth) & a.E.notna()].sort_values("E")
            ax[0, c].errorbar(r.E, r.F1_m, yerr=r.F1_se, marker="o", ms=3.5,
                              lw=1.3, color=C[mth], label=mth, capsize=2)
            ax[1, c].plot(r.E, r.active_m, marker="o", ms=3.5, lw=1.3,
                          color=C[mth], label=mth)
        for mth, ls in [("ADMM", "--"), ("Pooled-1SE", ":"),
                        ("Local-only-1SE", "-.")]:
            r = pick(a, s, mth)
            if r is None:
                continue
            ax[0, c].axhline(r.F1_m, ls=ls, lw=1.1, color=C[mth], label=mth)
            ax[1, c].axhline(r.active_m, ls=ls, lw=1.1, color=C[mth])
        ax[0, c].set_title(s, fontsize=10)
        ax[1, c].set_xlabel("Local epochs $E$")
        ax[1, c].set_xscale("log"); ax[1, c].set_xticks(E_GRID)
        ax[1, c].set_xticklabels(E_GRID)
    ax[0, 0].set_ylabel("$F_1$ vs.\\ true support")
    ax[1, 0].set_ylabel("Active coefficients")
    ax[0, 0].legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout(); fig.savefig(MS + "figures/fig1_support.pdf"); plt.close(fig)

    fig, ax = plt.subplots(1, 3, figsize=(9.4, 2.9))
    for c, s in enumerate(SC):
        r = a[(a.scenario == s) & (a.method == "FedAvg") & a.E.notna()].sort_values("E")
        ax[c].plot(r.E, r.vector_kib_m, marker="o", ms=3.5, lw=1.3,
                   color=C["FedAvg"], label="Upload (KiB)")
        a2 = ax[c].twinx()
        a2.plot(r.E, r.local_epochs_m, marker="s", ms=3.5, lw=1.3, ls="--",
                color=C["FedAvg-ST"], label="Local epochs")
        a2.grid(False)
        ax[c].set_title(s, fontsize=10)
        ax[c].set_xscale("log"); ax[c].set_xticks(E_GRID); ax[c].set_xticklabels(E_GRID)
        ax[c].set_xlabel("Local epochs $E$")
        if c == 0:
            ax[c].set_ylabel("Upload per fit (KiB)"); ax[c].legend(fontsize=7.5, frameon=False)
        if c == 2:
            a2.set_ylabel("Total CD epochs per site")
    fig.tight_layout(); fig.savefig(MS + "figures/fig2_cost.pdf"); plt.close(fig)

    d = pd.read_csv("real_data_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.8, 2.9))
    for mth in ["FedAvg", "FedAvg-ST", "FedAvg-P"]:
        r = d[d.method == mth].sort_values("E")
        ax[0].plot(r.E, r.active, marker="o", ms=3.5, lw=1.3, color=C[mth], label=mth)
        ax[1].plot(r.E, r.obj, marker="o", ms=3.5, lw=1.3, color=C[mth], label=mth)
    r = d[d.method == "ADMM"].iloc[0]
    ax[0].axhline(r.active, ls="--", lw=1.1, color=C["ADMM"], label="ADMM")
    ax[1].axhline(r.obj, ls="--", lw=1.1, color=C["ADMM"])
    for k in (0, 1):
        ax[k].set_xscale("log"); ax[k].set_xticks(E_GRID); ax[k].set_xticklabels(E_GRID)
        ax[k].set_xlabel("Local epochs $E$")
    ax[0].set_ylabel("Active coefficients"); ax[1].set_ylabel("Aggregated objective $F$")
    ax[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout(); fig.savefig(MS + "figures/fig3_realdata.pdf"); plt.close(fig)


def macros(a):
    d = pd.read_csv("real_data_results.csv").set_index(["method", "E"])
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
        m[f"M{tag}NRep"] = f"{int(p('FedAvg', 1).n)}"
        m[f"M{tag}RoundsOne"] = f"{p('FedAvg',1).rounds_m:.1f}"
        m[f"M{tag}RoundsTwenty"] = f"{p('FedAvg',20).rounds_m:.1f}"
        m[f"M{tag}EpochsOne"] = f"{p('FedAvg',1).local_epochs_m:.1f}"
        m[f"M{tag}EpochsTwenty"] = f"{p('FedAvg',20).local_epochs_m:.1f}"
        m[f"M{tag}AdmmRounds"] = f"{p('ADMM').rounds_m:.0f}"
        m[f"M{tag}ActDrop"] = f"{100*(1-p('FedAvg',20).active_m/p('FedAvg',1).active_m):.1f}\\%"
        m[f"M{tag}StScalar"] = f"{p('FedAvg-ST',1).scalar_kib_m:.1f}"
        m[f"M{tag}FedVectorOne"] = f"{p('FedAvg',1).vector_kib_m/p('FedAvg',1).rounds_m:.1f}"
    with open(MS + "numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))


if __name__ == "__main__":
    d = load(); d.to_csv("sim_all_raw.csv", index=False)
    a = agg(d); a.to_csv("sim_summary_new.csv", index=False)
    main_table(a); full_table(a); real_table(); eps_table(); figures(a); macros(a)
    for fp in glob.glob(MS + "tables/*.tex"):
        t = re.sub(r"(?<![A-Za-z])nan(?![A-Za-z])", "---", open(fp).read())
        open(fp, "w").write(t)
    cols = ["scenario","method","E","n","active_m","precision_m","recall_m",
            "F1_m","F1_se","coef_err_m","test_mse_m","rounds_m"]
    print(a[cols].round(3).to_string(index=False))
