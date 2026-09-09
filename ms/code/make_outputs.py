"""Build LaTeX tables, figures and \newcommand macros for the manuscript."""
import glob, numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MS = "/home/claude/work/ms/"
E_GRID = [1, 2, 3, 5, 8, 10, 15, 20]
SC = ["Independent", "Correlated", "Heterogeneous"]
REN = {"IID": "Independent"}

def load():
    fs = sorted(glob.glob("out_*.csv"))
    fs = [f for f in fs if "pilot" not in f]
    d = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    d["scenario"] = d["scenario"].replace(REN)
    return d

def agg(d, keys):
    g = d.groupby(keys, dropna=False)
    out = g.agg(n=("F1", "size"), **{f"{c}_m": (c, "mean") for c in
        ["active","TP","FP","precision","recall","F1","jaccard","coef_err",
         "test_mse","rounds","local_epochs","upload_kib"]},
        **{f"{c}_se": (c, lambda x: x.std(ddof=1)/np.sqrt(len(x))) for c in
           ["F1","jaccard","coef_err","test_mse","recall","precision","active"]})
    return g.size().to_frame("n").join(
        g[["active","TP","FP","precision","recall","F1","jaccard","coef_err",
           "test_mse","rounds","local_epochs","upload_kib"]].mean().add_suffix("_m")).join(
        g[["F1","jaccard","coef_err","test_mse","recall","precision","active"]]
         .sem().add_suffix("_se")).reset_index()

def f(x, k=3):
    return "---" if pd.isna(x) else f"{x:.{k}f}"

def main_table(a):
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Monte Carlo results over 100 replicates per design "
         r"($p=600$, $s=30$, $K=3$, $m=(400,240,160)$). Support metrics are "
         r"against the true support at $\epsilon=10^{-4}$. Monte Carlo "
         r"standard errors for $F_1$ in parentheses. \textsc{FedAvg} and "
         r"\textsc{FedAvg-ST} are shown at $E=1$ and $E=20$; the full grid "
         r"is in Table~\ref{tab:full}.}",
         r"\label{tab:main}",
         r"\begin{tabular}{llrrrrrrr}", r"\toprule",
         r"Design & Method & Active & Prec. & Recall & $F_1$ & "
         r"$\|\hat\beta-\beta^\star\|_2$ & Test MSE & Rounds \\", r"\midrule"]
    order = [("Pooled", np.nan), ("Local-only", np.nan), ("One-shot", np.nan),
             ("ADMM", np.nan), ("FedAvg", 1), ("FedAvg", 20),
             ("FedAvg-ST", 1), ("FedAvg-ST", 20)]
    for si, s in enumerate(SC):
        for i, (mth, E) in enumerate(order):
            r = a[(a.scenario == s) & (a.method == mth) &
                  ((a.E.isna()) if pd.isna(E) else (a.E == E))]
            if not len(r):
                continue
            r = r.iloc[0]
            nm = mth if pd.isna(E) else f"{mth} ($E={int(E)}$)"
            L.append(f"{s if i==0 else ''} & {nm} & {r.active_m:.1f} & "
                     f"{f(r.precision_m)} & {f(r.recall_m)} & "
                     f"{f(r.F1_m)} ({r.F1_se:.3f}) & {f(r.coef_err_m,2)} & "
                     f"{f(r.test_mse_m,2)} & {r.rounds_m:.1f} \\\\")
        L.append(r"\midrule" if si < len(SC) - 1 else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(MS + "tables/main_results.tex", "w").write("\n".join(L))

def full_table(a):
    L = [r"\begin{table}[t]", r"\centering", r"\footnotesize",
         r"\caption{Complete Monte Carlo results, 100 replicates per design. "
         r"MCSEs in parentheses. Upload is KiB of uploaded coefficient "
         r"vectors; epochs are total coordinate-descent passes per site.}",
         r"\label{tab:full}",
         r"\begin{tabular}{llrrrrrrrrr}", r"\toprule",
         r"Design & Method & $E$ & Active & Prec. & Recall & $F_1$ & Jacc. "
         r"& Rounds & Epochs & Upload \\", r"\midrule"]
    for si, s in enumerate(SC):
        sub = a[a.scenario == s]
        for mth in ["Pooled", "Local-only", "One-shot", "ADMM"]:
            r = sub[sub.method == mth]
            if not len(r): continue
            r = r.iloc[0]
            L.append(f"{s if mth=='Pooled' else ''} & {mth} & --- & {r.active_m:.1f} & "
                     f"{f(r.precision_m)} & {f(r.recall_m)} & {f(r.F1_m)} & "
                     f"{f(r.jaccard_m)} & {r.rounds_m:.1f} & --- & "
                     f"{r.upload_kib_m:.1f} \\\\")
        for mth in ["FedAvg", "FedAvg-ST"]:
            for E in E_GRID:
                r = sub[(sub.method == mth) & (sub.E == E)]
                if not len(r): continue
                r = r.iloc[0]
                L.append(f" & {mth} & {E} & {r.active_m:.1f} & {f(r.precision_m)} & "
                         f"{f(r.recall_m)} & {f(r.F1_m)} ({r.F1_se:.3f}) & "
                         f"{f(r.jaccard_m)} & {r.rounds_m:.1f} & "
                         f"{r.local_epochs_m:.1f} & {r.upload_kib_m:.1f} \\\\")
        L.append(r"\midrule" if si < len(SC) - 1 else "")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(MS + "tables/full_results.tex", "w").write("\n".join(L))

def real_table():
    d = pd.read_csv("real_data_results.csv")
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\caption{Three-site data illustration ($n=800$ training, $p=600$, "
         r"$K=3$). Agreement is with the pooled Lasso reference support "
         r"(\Srefactive\ active predictors), not with a known truth. $F$ is "
         r"the aggregated objective \eqref{eq:global}; consensus ADMM "
         r"minimises it exactly.}", r"\label{tab:real}",
         r"\begin{tabular}{lrrrrrrrrr}", r"\toprule",
         r"Method & $E$ & Active & TP & FP & Prec. & Recall & $F_1$ & $F$ "
         r"& Upload \\", r"\midrule"]
    for _, r in d.iterrows():
        E = "---" if pd.isna(r.E) else str(int(r.E))
        L.append(f"{r.method} & {E} & {int(r.active)} & {int(r.TP)} & "
                 f"{int(r.FP)} & {f(r.precision)} & {f(r.recall)} & "
                 f"{f(r.F1)} & {f(r.obj,3)} & {r.upload_kib:.1f} \\\\")
        if r.method == "ADMM":
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

C = {"FedAvg": "#1b4965", "FedAvg-ST": "#c1666b", "ADMM": "#4c956c",
     "Pooled": "#7a7a7a", "Local-only": "#b08968", "One-shot": "#9a86a4"}

def figures(a):
    plt.rcParams.update({"font.size": 9, "axes.grid": True,
                         "grid.alpha": 0.25, "axes.spines.top": False,
                         "axes.spines.right": False, "figure.dpi": 160})
    # Figure 1: F1 and active set vs E
    fig, ax = plt.subplots(2, 3, figsize=(9.2, 5.0), sharex=True)
    for c, s in enumerate(SC):
        sub = a[a.scenario == s]
        for mth in ["FedAvg", "FedAvg-ST"]:
            r = sub[(sub.method == mth) & sub.E.notna()].sort_values("E")
            ax[0, c].errorbar(r.E, r.F1_m, yerr=r.F1_se, marker="o", ms=3.5,
                              lw=1.3, color=C[mth], label=mth, capsize=2)
            ax[1, c].plot(r.E, r.active_m, marker="o", ms=3.5, lw=1.3,
                          color=C[mth], label=mth)
        for mth, ls in [("ADMM", "--"), ("Pooled", ":"), ("Local-only", "-.")]:
            r = sub[sub.method == mth]
            if not len(r): continue
            ax[0, c].axhline(r.F1_m.iloc[0], ls=ls, lw=1.1, color=C[mth], label=mth)
            ax[1, c].axhline(r.active_m.iloc[0], ls=ls, lw=1.1, color=C[mth])
        ax[0, c].set_title(s, fontsize=10)
        ax[1, c].set_xlabel("Local epochs $E$")
        ax[1, c].set_xscale("log"); ax[1, c].set_xticks(E_GRID)
        ax[1, c].set_xticklabels(E_GRID)
    ax[0, 0].set_ylabel("$F_1$ vs.\\ true support")
    ax[1, 0].set_ylabel("Active coefficients")
    ax[0, 0].legend(fontsize=7.5, frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(MS + "figures/fig1_support.pdf")
    plt.close(fig)

    # Figure 2: communication vs computation
    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for c, s in enumerate(SC):
        r = a[(a.scenario == s) & (a.method == "FedAvg") & a.E.notna()].sort_values("E")
        ax[c].plot(r.E, r.upload_kib_m, marker="o", ms=3.5, lw=1.3,
                   color=C["FedAvg"], label="Upload (KiB)")
        a2 = ax[c].twinx()
        a2.plot(r.E, r.local_epochs_m, marker="s", ms=3.5, lw=1.3, ls="--",
                color=C["FedAvg-ST"], label="Local epochs")
        a2.grid(False)
        ax[c].set_title(s, fontsize=10)
        ax[c].set_xscale("log"); ax[c].set_xticks(E_GRID); ax[c].set_xticklabels(E_GRID)
        ax[c].set_xlabel("Local epochs $E$")
        if c == 0: ax[c].set_ylabel("Upload per fit (KiB)")
        if c == 2: a2.set_ylabel("Total CD epochs per site")
    h1, l1 = ax[0].get_legend_handles_labels()
    ax[0].legend(fontsize=7.5, frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(MS + "figures/fig2_cost.pdf")
    plt.close(fig)

    # Figure 3: real data
    d = pd.read_csv("real_data_results.csv")
    fig, ax = plt.subplots(1, 2, figsize=(7.6, 2.9))
    for mth in ["FedAvg", "FedAvg-ST"]:
        r = d[d.method == mth].sort_values("E")
        ax[0].plot(r.E, r.active, marker="o", ms=3.5, lw=1.3, color=C[mth], label=mth)
        ax[1].plot(r.E, r.obj, marker="o", ms=3.5, lw=1.3, color=C[mth], label=mth)
    for mth, ls in [("ADMM", "--"), ("Pooled", ":")]:
        r = d[d.method == mth].iloc[0]
        ax[0].axhline(r.active, ls=ls, lw=1.1, color=C[mth], label=mth)
        ax[1].axhline(r.obj, ls=ls, lw=1.1, color=C[mth])
    for k in (0, 1):
        ax[k].set_xscale("log"); ax[k].set_xticks(E_GRID); ax[k].set_xticklabels(E_GRID)
        ax[k].set_xlabel("Local epochs $E$")
    ax[0].set_ylabel("Active coefficients")
    ax[1].set_ylabel("Aggregated objective $F$")
    ax[0].legend(fontsize=7.5, frameon=False)
    fig.tight_layout()
    fig.savefig(MS + "figures/fig3_realdata.pdf")
    plt.close(fig)

def macros(a):
    d = pd.read_csv("real_data_results.csv").set_index(["method", "E"])
    lam = open("log_real.txt").readlines()[1].split()
    m = {}
    fa1 = d.loc[("FedAvg", 1.0)]; fa20 = d.loc[("FedAvg", 20.0)]
    st1 = d.loc[("FedAvg-ST", 1.0)]
    adm = d.xs("ADMM", level=0).iloc[0]; loc = d.xs("Local-only", level=0).iloc[0]
    pool = d.xs("Pooled", level=0).iloc[0]
    m.update(Slamone="0.2046", Slamtwo="0.3635", Slamthree="0.1953",
             Slambar="0.2504", Slampooled="0.1778",
             Srefactive=f"{int(pool.active)}",
             Sfedactive=f"{int(fa1.active)}", SfedFone=f"{fa1.F1:.3f}",
             SlocalFone=f"{loc.F1:.3f}", Sstactive=f"{int(st1.active)}",
             SstFone=f"{st1.F1:.3f}", Sadmmactive=f"{int(adm.active)}",
             Sadmmtp=f"{int(adm.TP)}", Sadmmfp=f"{int(adm.FP)}",
             SadmmFone=f"{adm.F1:.3f}", Sadmmobj=f"{adm.obj:.3f}",
             Sfedobj=f"{fa1.obj:.3f}", Sfedobjtwenty=f"{fa20.obj:.3f}",
             Sgappct=f"{100*(fa1.obj-adm.obj)/adm.obj:.1f}\\%",
             Sroundsone=f"{int(fa1.rounds)}", Sroundstwenty=f"{int(fa20.rounds)}",
             Skibone=f"{fa1.upload_kib:.1f}", Skibtwenty=f"{fa20.upload_kib:.1f}",
             Sepochsone=f"{int(fa1.epochs)}", Sepochstwenty=f"{int(fa20.epochs)}",
             Sfedobjtwo=f"{d.loc[('FedAvg',2.0)].obj:.3f}",
             Sfedobjrange=f"{d.xs('FedAvg',level=0).obj.max()-d.xs('FedAvg',level=0).obj.min():.3f}",
             Sgapabs=f"{fa1.obj-adm.obj:.3f}")
    for s, tag in zip(SC, ["Ind", "Cor", "Het"]):
        sub = a[a.scenario == s]
        g = lambda mth, E=None: sub[(sub.method == mth) & ((sub.E.isna()) if E is None else (sub.E == E))].iloc[0]
        m[f"M{tag}FedActOne"] = f"{g('FedAvg',1).active_m:.0f}"
        m[f"M{tag}FedActTwenty"] = f"{g('FedAvg',20).active_m:.0f}"
        m[f"M{tag}FedFOne"] = f"{g('FedAvg',1).F1_m:.3f}"
        m[f"M{tag}FedFTwenty"] = f"{g('FedAvg',20).F1_m:.3f}"
        m[f"M{tag}StFOne"] = f"{g('FedAvg-ST',1).F1_m:.3f}"
        m[f"M{tag}StActOne"] = f"{g('FedAvg-ST',1).active_m:.0f}"
        m[f"M{tag}AdmmF"] = f"{g('ADMM').F1_m:.3f}"
        m[f"M{tag}AdmmAct"] = f"{g('ADMM').active_m:.0f}"
        m[f"M{tag}LocF"] = f"{g('Local-only').F1_m:.3f}"
        m[f"M{tag}PoolF"] = f"{g('Pooled').F1_m:.3f}"
        m[f"M{tag}RoundsOne"] = f"{g('FedAvg',1).rounds_m:.1f}"
        m[f"M{tag}RoundsTwenty"] = f"{g('FedAvg',20).rounds_m:.1f}"
        m[f"M{tag}EpochsOne"] = f"{g('FedAvg',1).local_epochs_m:.1f}"
        m[f"M{tag}EpochsTwenty"] = f"{g('FedAvg',20).local_epochs_m:.1f}"
        m[f"M{tag}NRep"] = f"{int(g('FedAvg',1).n)}"
        m[f"M{tag}FedPrecOne"] = f"{g('FedAvg',1).precision_m:.3f}"
        m[f"M{tag}FedRecOne"] = f"{g('FedAvg',1).recall_m:.3f}"
        m[f"M{tag}FedRecTwenty"] = f"{g('FedAvg',20).recall_m:.3f}"
        m[f"M{tag}OneShotF"] = f"{g('One-shot').F1_m:.3f}"
        m[f"M{tag}OneShotAct"] = f"{g('One-shot').active_m:.0f}"
        m[f"M{tag}StPrecOne"] = f"{g('FedAvg-ST',1).precision_m:.3f}"
        m[f"M{tag}StRecOne"] = f"{g('FedAvg-ST',1).recall_m:.3f}"
        m[f"M{tag}PoolAct"] = f"{g('Pooled').active_m:.0f}"
        m[f"M{tag}PoolErr"] = f"{g('Pooled').coef_err_m:.2f}"
        m[f"M{tag}PoolMse"] = f"{g('Pooled').test_mse_m:.1f}"
        m[f"M{tag}FedErrOne"] = f"{g('FedAvg',1).coef_err_m:.2f}"
        m[f"M{tag}FedMseOne"] = f"{g('FedAvg',1).test_mse_m:.1f}"
        m[f"M{tag}StErrOne"] = f"{g('FedAvg-ST',1).coef_err_m:.2f}"
        m[f"M{tag}StMseOne"] = f"{g('FedAvg-ST',1).test_mse_m:.1f}"
        m[f"M{tag}AdmmErr"] = f"{g('ADMM').coef_err_m:.2f}"
        m[f"M{tag}AdmmRounds"] = f"{g('ADMM').rounds_m:.0f}"
        m[f"M{tag}LocAct"] = f"{g('Local-only').active_m:.0f}"
        m[f"M{tag}ActDrop"] = f"{100*(1-g('FedAvg',20).active_m/g('FedAvg',1).active_m):.1f}\\%"
        m[f"M{tag}FedSeOne"] = f"{g('FedAvg',1).F1_se:.3f}"
        m[f"M{tag}StSeOne"] = f"{g('FedAvg-ST',1).F1_se:.3f}"
        m[f"M{tag}AdmmSe"] = f"{g('ADMM').F1_se:.3f}"
        m[f"M{tag}LocSe"] = f"{g('Local-only').F1_se:.3f}"
    with open(MS + "numbers.tex", "w") as fh:
        for k, v in m.items():
            fh.write("\\newcommand{\\%s}{%s}\n" % (k, v))

if __name__ == "__main__":
    d = load()
    d.to_csv("sim_all_raw.csv", index=False)
    a = agg(d, ["scenario", "method", "E"])
    a.to_csv("sim_summary_new.csv", index=False)
    main_table(a); full_table(a); real_table(); eps_table(); figures(a); macros(a)
    import re as _re, glob as _g
    for _f in _g.glob(MS + "tables/*.tex"):
        _t = open(_f).read()
        _t = _re.sub(r"(?<![A-Za-z])nan(?![A-Za-z])", "---", _t)
        open(_f, "w").write(_t)
    print(a[["scenario","method","E","n","active_m","precision_m","recall_m",
             "F1_m","F1_se","coef_err_m","test_mse_m","rounds_m",
             "local_epochs_m"]].round(3).to_string(index=False))
