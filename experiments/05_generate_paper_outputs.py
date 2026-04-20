"""
experiments/05_generate_paper_outputs.py
==========================================
Generate ALL paper figures and tables.

Current baseline = URD baseline:
  D = calibrated Mahalanobis on normalised residuals (sigma-temp scaled)
  U = sigma inflation ratio
  S = tuned FDE + run-length bonus
  combined = 0.35 * D_norm + 0.65 * S_norm   (weighted fusion, NOT max)

Comparison = TranAD (matched FD001 protocol, same sensors/split/window/healthy-only).

Run AFTER stages A / C / D / E:
  python -m experiments.01_train_baselines
  python -m experiments.02_synthetic_evaluation
  python -m experiments.02b_method_comparison
  python -m experiments.03_drift_classification
  python -m experiments.04_urd_fingerprinting
  python -m experiments.05_generate_paper_outputs

Outputs → outputs/for_paper/
"""

import csv, json, math, os, shutil, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from matplotlib.colors import LinearSegmentedColormap
import yaml
from scipy import stats as sp_stats

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PAPER = os.path.join(ROOT, "outputs", "for_paper")
os.makedirs(PAPER, exist_ok=True)
RES = os.path.join(ROOT, "outputs", "results")
FIG = os.path.join(ROOT, "outputs", "figures")

#─── palette (matches old project style) ────────────────────────────────────
PAL = {"blue":"#1565C0","orange":"#E65100","green":"#2E7D32","red":"#C62828",
       "purple":"#6A1B9A","gray":"#546E7A","teal":"#00695C","amber":"#F57F17",
       "urd":"#1565C0","tranad":"#E65100","nll":"#546E7A"}

METHOD_LINE = {
    "NLL":             dict(color="#546E7A", lw=1.3, ls="--"),
    "D+Conformity":    dict(color="#6A1B9A", lw=1.3, ls="--"),
    "D+Variance":      dict(color="#F57F17", lw=1.3, ls="--"),
    "D+FDE":           dict(color="#00695C", lw=1.3, ls="--"),
    "URD (baseline)":  dict(color="#1565C0", lw=2.8, ls="-"),
    "TranAD":          dict(color="#E65100", lw=2.0, ls="-."),
    "IForest":         dict(color="#AD1457", lw=1.3, ls=":")}

ANOM_TYPES = ["spike","drop","persistent_offset","noise_burst","sensor_freeze"]
DRIFT_TYPES = ["gradual_shift","sigmoid_plateau","accelerating","multi_sensor"]
ALL_TYPES = ANOM_TYPES + DRIFT_TYPES
CAT_MAP = {"spike":"point_anomaly","drop":"point_anomaly","persistent_offset":"persistent_shift",
           "noise_burst":"noise_anomaly","sensor_freeze":"sensor_malfunction",
           "gradual_shift":"drift","sigmoid_plateau":"drift","accelerating":"drift","multi_sensor":"drift"}

#─── latest confirmed results ────────────────────────────────────────────────
FREEZE_ROC = {"URD (baseline)":0.8230,"TranAD":0.4621,"NLL":0.4398}
STAGE_C_OVERALL = {"URD (baseline)":(0.8636,0.4250),"TranAD":(0.7379,0.2475)}


def _style():
    plt.rcParams.update({
        "figure.facecolor":"white","axes.facecolor":"#F9F9F9",
        "savefig.facecolor":"white","font.family":"DejaVu Sans",
        "font.size":11,"axes.labelsize":12,"axes.titlesize":13,
        "xtick.labelsize":10,"ytick.labelsize":10,
        "legend.fontsize":9,"legend.framealpha":0.92,
        "lines.linewidth":1.8,"axes.linewidth":0.8,
        "axes.grid":True,"grid.alpha":0.22,"grid.linewidth":0.55,
        "axes.spines.top":False,"axes.spines.right":False})


def _save(fig, name):
    p = os.path.join(PAPER, name)
    fig.savefig(p, dpi=200, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"  saved  {name}")


def _csv_write(rows, headers, name):
    p = os.path.join(PAPER, name)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(headers)
        for r in rows: w.writerow(r)
    print(f"  saved  {name}")


def _cfg():
    with open(os.path.join(ROOT, "config", "config.yaml")) as f:
        return yaml.safe_load(f)


def _load_json(path):
    if not os.path.exists(path): return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_csv_rows(path):
    if not os.path.exists(path): return [], []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)
    return rows, r.fieldnames if rows else []


def _load_data(cfg):
    from src.data.loader import load_train_data
    from src.data.preprocessing import compute_life_fraction, select_sensors, SensorScaler
    from src.data.splits import split_engines, apply_split
    sensors = cfg["dataset"]["selected_sensors"]
    df = load_train_data(cfg["paths"]["raw_data_dir"], cfg["dataset"]["subset"])
    df = compute_life_fraction(df); df = select_sensors(df, sensors, keep_meta=True)
    ti,vi,si = split_engines(df, cfg["preprocessing"]["train_ratio"], cfg["preprocessing"]["val_ratio"], cfg["preprocessing"]["test_ratio"], cfg["preprocessing"]["split_random_seed"])
    sp = apply_split(df, ti, vi, si); sc = SensorScaler(sensors)
    sp["train"]=sc.fit_transform(sp["train"]); sp["val"]=sc.transform(sp["val"]); sp["test"]=sc.transform(sp["test"])
    return sp, sensors


def _load_gru(cfg, device="cpu"):
    import torch
    from src.models.gaussian_gru import GaussianGRU
    m = GaussianGRU(input_size=len(cfg["dataset"]["selected_sensors"]), hidden_size=cfg["model"]["hidden_size"], num_layers=cfg["model"]["num_layers"], dropout=0.0, sigma_min=cfg["model"]["sigma_min"])
    ckpt = os.path.join(cfg["paths"]["model_dir"], "gaussian_gru_best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval(); return m


def _load_tranad(cfg, device="cpu"):
    """Load TranAD checkpoint. Returns None gracefully if missing."""
    try:
        import torch
        from src.models.tranad import TranAD
        tcfg = cfg["model"].get("tranad", {})
        m = TranAD(input_size=len(cfg["dataset"]["selected_sensors"]), window_size=cfg["preprocessing"]["window_size"], d_model=tcfg.get("d_model",64), nhead=tcfg.get("nhead",4), num_layers=tcfg.get("num_layers",2), dim_feedforward=tcfg.get("dim_feedforward",128), dropout=tcfg.get("dropout",0.1))
        ckpt = os.path.join(cfg["paths"]["model_dir"], "tranad_best.pt")
        if not os.path.exists(ckpt): return None
        m.load_state_dict(torch.load(ckpt, map_location=device)); m.eval(); return m
    except Exception:
        return None


def _infer_gru(model, values, W, device="cpu"):
    import torch
    T, d = values.shape
    if T <= W: return None, None, None
    X = np.array([values[i:i+W] for i in range(T-W)], dtype=np.float32)
    y = np.array([values[i+W] for i in range(T-W)], dtype=np.float32)
    with torch.no_grad():
        mu, sigma = model(torch.tensor(X).to(device))
    return y, mu.cpu().numpy(), sigma.cpu().numpy()


def _infer_tranad(model, values, W, device="cpu"):
    import torch
    T, d = values.shape
    if T <= W: return None, None, None
    X = np.array([values[i:i+W] for i in range(T-W)], dtype=np.float32)
    y = np.array([values[i+W] for i in range(T-W)], dtype=np.float32)
    with torch.no_grad():
        p1, p2 = model(torch.tensor(X).to(device))
    return y, p1.cpu().numpy(), p2.cpu().numpy()


def _fit_urd(model, val_df, sensors, W, device="cpu"):
    from src.anomaly.urd import URDScorer
    urd = URDScorer(fde_window=5); ys,ms,ss=[],[],[]
    for eid in sorted(val_df["unit_nr"].unique()):
        v = val_df[val_df["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
        y,mu,sg = _infer_gru(model, v, W, device)
        if y is not None: ys.append(y); ms.append(mu); ss.append(sg)
    if ys: urd.fit(np.concatenate(ys), np.concatenate(ms), np.concatenate(ss))
    return urd


def _fit_nll(model, val_df, sensors, W, device="cpu"):
    from src.anomaly.scoring import AnomalyScorer
    sc = AnomalyScorer(score_type="nll"); ys,ms,ss=[],[],[]
    for eid in sorted(val_df["unit_nr"].unique()):
        v = val_df[val_df["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
        y,mu,sg = _infer_gru(model, v, W, device)
        if y is not None: ys.append(y); ms.append(mu); ss.append(sg)
    if ys: sc.fit_normalization(np.concatenate(ys), np.concatenate(ms), np.concatenate(ss))
    return sc


def _ed(df, eid, sensors):
    edf = df[df["unit_nr"]==eid].sort_values("time_cycles")
    return {"engine_id":int(eid),"sensor_values":edf[sensors].values.copy(),"cycles":edf["time_cycles"].values.copy(),"life_fracs":edf["life_fraction"].values.copy()}


def _inject_drift(base, dtype, T):
    dv=base.copy(); ds=T//2
    for t in range(ds, T):
        p=(t-ds)/max(1,T-ds)
        if dtype=="gradual_shift": dv[t,0]+=p*3.0
        elif dtype=="sigmoid_plateau": dv[t,0]+=3.0/(1+np.exp(-10*(p-0.5)))
        elif dtype=="accelerating": dv[t,0]+=p**2*4.0
        elif dtype=="multi_sensor": dv[t,0]+=p*2.0; dv[t,min(2,dv.shape[1]-1)]+=p*1.5
    return dv, ds


def _tranad_score(ty, p1, p2, val_mse_mu=None, val_mse_std=None):
    err = np.mean((ty - p2)**2, axis=1)
    if val_mse_mu is not None and val_mse_std is not None:
        return (err - val_mse_mu) / max(val_mse_std, 1e-8)
    mu = float(np.mean(err[:max(5,len(err)//3)])); sd = max(float(np.std(err[:max(5,len(err)//3)])),1e-8)
    return (err - mu) / sd


def _nll_score(y, mu, sigma, nll_sc):
    s, _ = nll_sc.score(y, mu, sigma, normalize=True)
    return s


#═══════════════════════════════════════════════════════════════════════════════
#FIG 1 — Pipeline Overview
#═══════════════════════════════════════════════════════════════════════════════
def fig1_pipeline_overview():
    _style()
    fig, ax = plt.subplots(figsize=(20, 9))
    ax.set_xlim(0, 22); ax.set_ylim(0, 9); ax.axis("off")

    BOXES = [
        (0.2, 4.4, 2.6, 2.8, "STEP 1\nRaw Data", "train_FD001.txt\n100 engines\n26 cols per row\nwhitespace-delim\nno header", "#E3F2FD"),
        (3.2, 4.4, 2.6, 2.8, "STEP 2\nPreprocessing", "21→7 sensors\nengine split\nwindow = 30\nZ-score scale\nhealthy-only train", "#E8F5E9"),
        (6.2, 4.4, 2.8, 2.8, "STEP 3\nGaussian GRU", "Input: (30, 7)\n→ μ ∈ ℝ⁷\n→ σ ∈ ℝ⁷\ntrained with NLL\nlife_frac ≤ 0.5", "#FFF3E0"),
        (9.5, 4.4, 3.2, 2.8, "STEP 4\nURD Baseline", "D = calib. Mahalanobis\nU = sigma inflation\nS = tuned FDE + run\nA = 0.35D + 0.65S", "#FCE4EC"),
        (13.2, 4.4, 3.0, 2.8, "STEP 5\nTranAD Baseline", "two-phase transformer\nself-conditioning\nnext-step error score\nsame FD001 protocol", "#E8F5E9"),
        (16.8, 6.0, 2.8, 1.6, "STEP 6a\nClassify", "drift vs anomaly\n16-feature URD model\nRF acc = 0.955", "#EDE7F6"),
        (16.8, 4.1, 2.8, 1.7, "STEP 6b\nFingerprint", "5-class type ID\nacc = 0.900\n9-class acc = 0.631", "#E8EAF6")]

    for bx,by,bw,bh,title,body,col in BOXES:
        ax.add_patch(mpatches.FancyBboxPatch((bx,by),bw,bh,boxstyle="round,pad=0.12",facecolor=col,edgecolor="#546E7A",linewidth=1.5))
        ax.text(bx+bw/2,by+bh-0.28,title,ha="center",va="top",fontsize=9.5,fontweight="bold",color="#1A237E")
        ax.text(bx+bw/2,by+0.18,body,ha="center",va="bottom",fontsize=8.1,color="#263238",linespacing=1.55)

    akw = dict(arrowstyle="-|>",color="#37474F",lw=1.8)
    for x1,x2,yc in [(2.8,3.2,5.8),(5.8,6.2,5.8),(9.0,9.5,5.8),(12.7,13.2,5.8),(16.2,16.8,6.8),(16.2,16.8,4.95)]:
        ax.annotate("",xy=(x2,yc),xytext=(x1,yc),arrowprops=dict(**akw))
    ax.annotate("",xy=(16.8,6.8),xytext=(16.2,7.0),arrowprops=dict(arrowstyle="-|>",color="#37474F",lw=1.3))
    ax.annotate("",xy=(16.8,4.95),xytext=(16.2,4.7),arrowprops=dict(arrowstyle="-|>",color="#37474F",lw=1.3))

    MATH = [
        (1.5,4.2,r"$\mathbf{x}_t\in\mathbb{R}^{21}$"+"\nunit_nr, time_cycles\n21 raw sensors"),
        (4.5,4.2,r"$z_j=(x_j-\bar{x}_j^{tr})/s_j^{tr}$"+"\nfit scaler on train only\nengine-level split"),
        (7.6,4.2,r"$\mathcal{L}=\frac{1}{d}\sum_j[\log\sigma_j+\frac{(x_j-\mu_j)^2}{2\sigma_j^2}]$"+"\nprobabilistic next-step"),
        (11.1,4.2,r"$r_t=(\frac{x-\mu}{\tau\odot\sigma})$, $D_t=r_t^\top\Sigma_r^{-1}r_t$"+"\n"+r"$S_t=S_t^{fde}+3\max(0,run_t-1)$"+"\n"+r"$A_t=0.35\tilde{D}_t+0.65\tilde{S}_t$"),
        (14.7,4.2,"TranAD two-phase score\n$score_t=||y_t-\\hat{y}_t^{(2)}||^2$\nsame healthy-only train"),
        (18.2,4.2,"drift vs anomaly\n5-class fingerprint\ninterpretable outputs")]
    for x,y,txt in MATH:
        ax.text(x,y,txt,ha="center",va="top",fontsize=7.6,color="#37474F",style="italic",linespacing=1.5,bbox=dict(boxstyle="round,pad=0.28",facecolor="#FAFAFA",alpha=0.88,edgecolor="#B0BEC5"))

    ax.text(11.0,8.6,"Matched comparison: FD001 · same 7 sensors · window=30 · engine split · healthy-only training · same synthetic evaluation suite",ha="center",fontsize=9.5,color=PAL["blue"],fontweight="bold",bbox=dict(boxstyle="round,pad=0.35",facecolor="#E3F2FD",edgecolor=PAL["blue"],alpha=0.92))
    ax.set_title("Project Pipeline — URD Baseline vs TranAD Comparison\nPrimary model: Gaussian GRU  |  URD: calibrated Mahalanobis D + tuned FDE/run S + weighted fusion",fontsize=13.5,fontweight="bold",pad=10)
    _save(fig, "fig1_pipeline_overview.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG 2 — URD Channel Visualisation (updated baseline math)
#═══════════════════════════════════════════════════════════════════════════════
def fig2_urd_channels(model, urd, test_df, sensors, W, device="cpu"):
    _style()
    scenarios = []
    for eid in sorted(test_df["unit_nr"].unique())[:8]:
        ed = _ed(test_df, eid, sensors); base=ed["sensor_values"]; T=len(base); lf=ed["life_fracs"]
        if T < 90: continue
        mid = T // 3
        sv = base.copy(); sv[mid,0]+=6.5; sv[mid+1,0]+=3.0
        scenarios.append(("(a)  Spike Anomaly — D fires strongly", sv, lf, PAL["red"]))
        dv,_ = _inject_drift(base,"gradual_shift",T)
        scenarios.append(("(b)  Gradual Drift — U inflates", dv, lf, PAL["green"]))
        fv = base.copy(); fv[mid:mid+45,1]=fv[mid,1]
        scenarios.append(("(c)  Sensor Freeze — S fires (FDE+run)", fv, lf, PAL["blue"]))
        break
    if not scenarios: print("  skipped fig2 — trajectories too short"); return

    ch_labels = [
        r"$D_t = r_t^\top \Sigma_r^{-1} r_t$   (calibrated Mahalanobis deviation)",
        r"$U_t = \frac{1}{d}\sum_j \sigma^{eff}_{t,j}/\sigma^{ref}_j$   (sigma inflation)",
        r"$S_t = S_t^{fde}+3\max(0,run_t-1)$   (tuned FDE + run-length)"]
    ch_colors = [PAL["red"], PAL["orange"], PAL["blue"]]

    fig, axes = plt.subplots(3, 3, figsize=(16, 9), sharex="col")
    for row,(title,vals,lf,tc) in enumerate(scenarios):
        y,mu,sg = _infer_gru(model, vals, W, device)
        if y is None: continue
        res = urd.score(y, mu, sg)
        n = len(res["deviation"]); lf_p = lf[W:W+n]
        chs = [res["deviation"], res["uncertainty"], res["stationarity"]]
        for col in range(3):
            ax = axes[row,col]; ch = np.nan_to_num(chs[col], nan=0.0)
            if col != 1: ch = np.clip(ch, 0, None)
            ax.plot(lf_p, ch, color=ch_colors[col], lw=1.9, alpha=0.92)
            base_val = 1.0 if col==1 else 0.0
            ax.fill_between(lf_p, base_val, ch, alpha=0.13, color=ch_colors[col])
            if col==1: ax.axhline(1.0, color=PAL["gray"], ls="--", lw=0.9, alpha=0.6, label="U=1 (baseline)"); ax.legend(fontsize=8,loc="upper left")
            pk = int(np.argmax(np.abs(ch-base_val)))
            if abs(ch.max()-base_val) > 0.1: ax.annotate(f"  peak={ch[pk]:.2f}", xy=(lf_p[pk],ch[pk]), fontsize=8, color=ch_colors[col], fontweight="bold")
            if row==0: ax.set_title(ch_labels[col], fontsize=9.5, fontweight="bold", pad=7)
            if col==0: ax.set_ylabel(title, fontsize=9.5, color=tc, fontweight="bold", labelpad=8)
            if row==2: ax.set_xlabel("Life Fraction", fontsize=10)
            ax.tick_params(labelsize=8.5)

    fig.suptitle("URD Three-Channel Decomposition — Current Baseline\n"
                 r"Calibrated residual: $r_{t,j}=(x_{t,j}-\mu_{t,j})/(\tau_j\sigma_{t,j})$  "
                 r"|  Final score: $A_t=0.35\tilde{D}_t+0.65\tilde{S}_t$",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout(); _save(fig, "fig2_urd_channels.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG 3 — Sensor Freeze Blind Spot (updated numbers and methods)
#═══════════════════════════════════════════════════════════════════════════════
def fig3_sensor_freeze_blind_spot(model, urd, nll_sc, test_df, sensors, W, tranad_model=None, device="cpu"):
    _style()
    for eid in sorted(test_df["unit_nr"].unique()):
        ed = _ed(test_df, eid, sensors); base=ed["sensor_values"]; lf=ed["life_fracs"]; T=len(base)
        if T < 100: continue
        fs, fe = T//4, 3*T//4
        norm_v = base.copy(); frz_v = base.copy(); frz_v[fs:fe,1] = frz_v[fs,1]
        y_n,mu_n,sg_n = _infer_gru(model, norm_v, W, device)
        y_f,mu_f,sg_f = _infer_gru(model, frz_v, W, device)
        if y_n is None or y_f is None: continue
        res_n = urd.score(y_n, mu_n, sg_n); res_f = urd.score(y_f, mu_f, sg_f)
        nll_n,_ = nll_sc.score(y_n, mu_n, sg_n, normalize=True)
        nll_f,_ = nll_sc.score(y_f, mu_f, sg_f, normalize=True)
        n = min(len(nll_n), len(nll_f), len(res_n["stationarity"]))
        lf_p = lf[W:W+n]; fs_lf = lf[min(fs,T-1)]; fe_lf = lf[min(fe,T-1)]

        #Compute TranAD scores if available
        tranad_n = tranad_f = None
        if tranad_model is not None:
            try:
                ty_n,tp1_n,tp2_n = _infer_tranad(tranad_model, norm_v, W, device)
                ty_f,tp1_f,tp2_f = _infer_tranad(tranad_model, frz_v, W, device)
                if ty_n is not None:
                    #fit normalisation on normal predictions
                    val_err = np.mean((ty_n - tp2_n)**2, axis=1)
                    v_mu = float(np.mean(val_err[:max(5,n//3)])); v_sd = max(float(np.std(val_err[:max(5,n//3)])), 1e-8)
                    tranad_n = _tranad_score(ty_n[:n], tp1_n[:n], tp2_n[:n], v_mu, v_sd)
                    tranad_f = _tranad_score(ty_f[:n], tp1_f[:n], tp2_f[:n], v_mu, v_sd)
            except Exception:
                pass

        n_panels = 3 if tranad_n is not None else 2
        fig, axes = plt.subplots(1, n_panels, figsize=(7*n_panels, 5.5))
        if n_panels == 2: axes = list(axes)

        panels = [
            (axes[0], nll_n[:n], nll_f[:n], "NLL Anomaly Score (normalised)",
             "(a)  NLL — BLIND to sensor freeze",
             f"Freeze scored as 'extra normal'\nROC-AUC = {FREEZE_ROC['NLL']:.4f}  (sub-random!)", PAL["gray"]),
            (axes[1], np.clip(res_n["combined"][:n],0,None), np.clip(res_f["combined"][:n],0,None),
             r"URD baseline score  $A_t=0.35\tilde{D}+0.65\tilde{S}$",
             "(b)  URD baseline — DETECTS sensor freeze",
             f"Calibrated Mahalanobis D + tuned FDE/run S\nROC-AUC = {FREEZE_ROC['URD (baseline)']:.4f}  (+{FREEZE_ROC['URD (baseline)']-FREEZE_ROC['NLL']:.4f} vs NLL)", PAL["blue"])]
        if tranad_n is not None:
            panels.append((axes[2], tranad_n, tranad_f, "TranAD anomaly score (normalised)",
                           "(c)  TranAD — residual-only, also struggles with freeze",
                           f"Transformer next-step error — freeze looks 'easy to predict'\nROC-AUC = {FREEZE_ROC['TranAD']:.4f}  (similar to NLL)", PAL["orange"]))

        for ax,yn,yf,ylabel,ttl,note,c in panels:
            ax.plot(lf_p, yn, color=PAL["green"], lw=1.7, label="Normal engine", alpha=0.9)
            ax.plot(lf_p, yf, color=c, lw=1.7, label="Frozen sensor (s_4)", alpha=0.92)
            ax.axvspan(fs_lf, fe_lf, alpha=0.07, color=c)
            ax.axvline(fs_lf, color=c, ls="--", lw=1.1, alpha=0.65, label="Freeze window")
            ax.axvline(fe_lf, color=c, ls="--", lw=1.1, alpha=0.65)
            ax.set_xlabel("Life Fraction", fontweight="bold")
            ax.set_ylabel(ylabel, fontweight="bold")
            ax.set_title(ttl, fontweight="bold", color=c, pad=10)
            ax.legend(loc="upper left", fontsize=9)
            ax.text(0.5, 0.06, note, transform=ax.transAxes, ha="center", fontsize=9, color=c, style="italic", bbox=dict(boxstyle="round,pad=0.35",facecolor="white",alpha=0.93,edgecolor=c))

        fig.suptitle(f"Sensor Freeze Blind Spot — Why Residual-Only Detectors Fail\n"
                     f"NLL ROC={FREEZE_ROC['NLL']:.4f}  |  TranAD ROC={FREEZE_ROC['TranAD']:.4f}  |  URD baseline ROC={FREEZE_ROC['URD (baseline)']:.4f}",
                     fontsize=12.5, fontweight="bold")
        plt.tight_layout(); _save(fig, "fig3_sensor_freeze_blind_spot.png"); break


#═══════════════════════════════════════════════════════════════════════════════
#FIG — ROC/PR direct comparison: URD vs TranAD (restyled to old look)
#═══════════════════════════════════════════════════════════════════════════════
def fig_roc_pr_urd_vs_tranad(stage_c):
    _style()
    if not stage_c: print("  skipped fig_roc_pr — no Stage C results"); return
    urd_data = stage_c.get("urd",{}); tranad_data = stage_c.get("tranad",{})
    if not urd_data or not tranad_data: print("  skipped fig_roc_pr — missing urd/tranad keys"); return

    urd_roc, urd_pr = STAGE_C_OVERALL["URD (baseline)"]
    tr_roc, tr_pr = STAGE_C_OVERALL["TranAD"]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, key, xlabel, ylabel, ttl in [
        (axes[0],"roc","False Positive Rate","True Positive Rate",f"ROC Curve — URD vs TranAD"),
        (axes[1],"pr","Recall","Precision",f"Precision-Recall — URD vs TranAD")]:
        for label, data, c, lw, ls, auc in [
            ("URD baseline", urd_data, PAL["blue"], 2.4, "-", urd_roc if key=="roc" else urd_pr),
            ("TranAD", tranad_data, PAL["orange"], 1.8, "-.", tr_roc if key=="roc" else tr_pr)]:
            curves = data.get("curves",{}).get(key,[])
            if len(curves) == 2 and len(curves[0]) > 1:
                x, y = curves
                ax.plot(x, y, color=c, lw=lw, ls=ls, label=f"{label} ({auc:.3f})", alpha=0.92)
        if key == "roc": ax.plot([0,1],[0,1],"k--",lw=0.7,alpha=0.25)
        ax.set_xlabel(xlabel, fontweight="bold"); ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(ttl, fontweight="bold", pad=8)
        ax.legend(fontsize=10, loc="lower right" if key=="roc" else "upper right")
        if key == "roc": ax.set_xlim([0,1]); ax.set_ylim([0,1.02])

    fig.suptitle("Direct Detector Comparison on Synthetic FD001 Test Suite\n"
                 "Same engines · same 7 sensors · window=30 · healthy-only training · same 1080 trajectories",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig2_roc_pr_urd_vs_tranad.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG — Threshold sweep (restyled to old look)
#═══════════════════════════════════════════════════════════════════════════════
def fig_threshold_sweep():
    _style()
    sweep_path = os.path.join(RES, "stage_c_threshold_sweep.csv")
    thr_path = os.path.join(RES, "stage_c_threshold_metrics.csv")
    source = sweep_path if os.path.exists(sweep_path) else (thr_path if os.path.exists(thr_path) else None)
    if source is None: print("  skipped fig_threshold_sweep — no CSV found"); return

    rows,_ = _load_csv_rows(source)
    if not rows: print("  skipped fig_threshold_sweep — empty CSV"); return

    #Infer column names
    sample = rows[0]
    has_threshold = "threshold" in sample
    has_pct = "threshold_pct" in sample
    if not has_threshold and not has_pct: print("  skipped threshold sweep — unknown format"); return

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
    metric_keys = [("precision","Precision"),("recall","Recall"),("f1","F1 Score")]
    for method, color, lw, ls in [("URD",PAL["blue"],2.0,"-"),("TranAD",PAL["orange"],1.6,"-.")]:
        sub = [r for r in rows if r.get("method","") == method]
        if not sub: continue
        thr_col = "threshold" if has_threshold else "threshold_pct"
        try:
            thr = [float(r[thr_col]) for r in sub]
        except Exception:
            continue
        for ax,(mk,mlabel) in zip(axes, metric_keys):
            try:
                vals = [float(r.get(mk, r.get(mk.replace("f1","f1_score"),0))) for r in sub]
            except Exception:
                continue
            ax.plot(thr, vals, color=color, lw=lw, ls=ls, label=method, alpha=0.92)

    #Annotate p95 / p97.5 / p99 vertical guides
    for ax in axes:
        for pct, lbl in [(95,"p95"),(97.5,"p97.5"),(99,"p99")]:
            ax.axvline(pct, color=PAL["gray"], ls=":", lw=0.9, alpha=0.5)
            ax.text(pct+0.15, ax.get_ylim()[0] if ax.get_ylim()[0]>0 else 0.02, lbl, fontsize=7.5, color=PAL["gray"], rotation=90, va="bottom")

    for ax,(mk,mlabel) in zip(axes, metric_keys):
        ax.set_title(mlabel, fontweight="bold", pad=8)
        ax.set_xlabel("Threshold Percentile", fontweight="bold")
        ax.legend(fontsize=9)
    axes[0].set_ylabel("Metric Value", fontweight="bold")
    fig.suptitle("Threshold Sweep — Point-Level Detection Trade-offs\nURD baseline vs TranAD across percentile thresholds",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig3_threshold_sweep.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG — Per-type PR bar chart (new, styled to old project look)
#═══════════════════════════════════════════════════════════════════════════════
def fig_per_type_pr(stage_c):
    _style()
    if not stage_c: print("  skipped fig_per_type_pr — no Stage C results"); return
    urd_pt = stage_c.get("urd",{}).get("per_type",{})
    tr_pt = stage_c.get("tranad",{}).get("per_type",{})
    if not urd_pt: print("  skipped fig_per_type_pr — no per_type data"); return

    types = ANOM_TYPES
    x = np.arange(len(types))
    w = 0.35
    urd_pr = [urd_pt.get(t,{}).get("pr_auc",0) for t in types]
    tr_pr  = [tr_pt.get(t,{}).get("pr_auc",0) for t in types]

    fig, ax = plt.subplots(figsize=(11, 5.5))
    bars_u = ax.bar(x-w/2, urd_pr, w, color=PAL["blue"], alpha=0.88, label="URD baseline", edgecolor="white")
    bars_t = ax.bar(x+w/2, tr_pr, w, color=PAL["orange"], alpha=0.88, label="TranAD", edgecolor="white")
    for bar in bars_u:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=PAL["blue"], fontweight="bold")
    for bar in bars_t:
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.01, f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8, color=PAL["orange"])

    ax.set_xticks(x)
    ax.set_xticklabels([t.replace("_"," ").title() for t in types], fontsize=10.5, fontweight="bold")
    ax.set_ylabel("PR-AUC", fontweight="bold", fontsize=12)
    ax.set_ylim(0, min(1.08, max(max(urd_pr), max(tr_pr)) + 0.18))
    ax.set_title("Per-Anomaly-Type PR-AUC — URD Baseline vs TranAD\nSensor Freeze is the defining blind spot for residual-only methods",
                 fontweight="bold", fontsize=12.5, pad=10)
    ax.legend(fontsize=10)
    #Highlight freeze column
    freeze_idx = types.index("sensor_freeze")
    ax.axvspan(freeze_idx-0.5, freeze_idx+0.5, alpha=0.06, color=PAL["blue"], zorder=0)
    plt.tight_layout(); _save(fig, "fig4_per_type_pr.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG — ROC curves per type: ALL 7 METHODS
#═══════════════════════════════════════════════════════════════════════════════
def fig4_roc_curves_by_type(cfg, splits, sensors, W, model, urd, nll_sc, tranad_model=None, device="cpu"):
    from sklearn.metrics import roc_curve, roc_auc_score
    from src.synthetic.anomaly_generator import AnomalyGenerator
    from scipy import stats
    _style()
    gen = AnomalyGenerator(sensors, random_seed=42)
    eids = sorted(splits["test"]["unit_nr"].unique())[:10]

    #Fit method-specific scorers on validation data
    import warnings; warnings.filterwarnings("ignore")
    Xv = []; yv = []; muv = []; sgv = []
    for eid in sorted(splits["val"]["unit_nr"].unique()):
        vals = splits["val"][splits["val"]["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
        y,mu,sg = _infer_gru(model,vals,W,device)
        if y is None: continue
        Xv.append(vals[W:W+len(y)]); yv.append(y); muv.append(mu); sgv.append(sg)
    if not yv: print("  skipped fig4_roc_curves_by_type — no val data"); return
    val_y = np.concatenate(yv); val_mu = np.concatenate(muv); val_sg = np.concatenate(sgv); val_raw = np.concatenate(Xv)

    #Build all scorers
    class ConformityScorer:
        def __init__(self, window=10): self.w=window; self.d_mn=self.d_sd=self.c_mn=self.c_sd=None
        def fit(self,y,mu,sg):
            d=np.mean(((y-mu)/sg)**2,axis=1); self.d_mn,self.d_sd=d.mean(),max(d.std(),1e-8)
            c=self._conf(y,mu,sg); v=c[~np.isnan(c)]; self.c_mn,self.c_sd=v.mean(),max(v.std(),1e-8)
        def _conf(self,y,mu,sg):
            T,d=y.shape; w=self.w; zs=((y-mu)/sg)**2; sc=np.full(T,np.nan)
            for t in range(w-1,T):
                q=np.sum(zs[t-w+1:t+1],axis=0); pv=stats.chi2.cdf(q,df=w); sc[t]=max(0,-np.log(np.min(pv)+1e-15))
            return sc
        def score(self,y,mu,sg,raw=None):
            d=(np.mean(((y-mu)/sg)**2,axis=1)-self.d_mn)/self.d_sd
            c=self._conf(y,mu,sg); c=np.nan_to_num((c-self.c_mn)/self.c_sd,nan=0.0)
            return np.maximum(d,c)

    class VarianceScorer:
        def __init__(self, w=10, eps=1e-10): self.w=w; self.eps=eps; self.vr=None; self.d_mn=self.d_sd=self.s_mn=self.s_sd=None
        def fit(self,y,mu,sg):
            self.vr=np.maximum(np.var(y,axis=0),self.eps)
            d=np.mean(((y-mu)/sg)**2,axis=1); self.d_mn,self.d_sd=d.mean(),max(d.std(),1e-8)
            ss=self._stat(y); v=ss[~np.isnan(ss)]; self.s_mn,self.s_sd=v.mean(),max(v.std(),1e-8)
        def _stat(self,raw):
            T,d=raw.shape; w=self.w; sc=np.full(T,np.nan)
            for t in range(w-1,T):
                v=np.var(raw[t-w+1:t+1],axis=0); r=v/self.vr; sc[t]=np.max(np.maximum(-np.log(r+self.eps),0.0))
            return sc
        def score(self,y,mu,sg,raw=None):
            if raw is None: raw=y
            d=(np.mean(((y-mu)/sg)**2,axis=1)-self.d_mn)/self.d_sd
            s=self._stat(raw); s=np.nan_to_num((s-self.s_mn)/self.s_sd,nan=0.0)
            return np.maximum(d,s)

    class FDEScorer:
        def __init__(self, w=5, eps=1e-10): self.w=w; self.eps=eps; self.fr=None; self.d_mn=self.d_sd=self.s_mn=self.s_sd=None
        def fit(self,y,mu,sg):
            diffs=np.diff(y,axis=0); self.fr=np.maximum(np.mean(diffs**2,axis=0),self.eps)
            d=np.mean(((y-mu)/sg)**2,axis=1); self.d_mn,self.d_sd=d.mean(),max(d.std(),1e-8)
            ss=self._stat(y); v=ss[~np.isnan(ss)]; self.s_mn,self.s_sd=v.mean(),max(v.std(),1e-8)
        def _stat(self,raw):
            T,d=raw.shape; w=self.w; diffs=np.zeros_like(raw); diffs[1:]=raw[1:]-raw[:-1]; sq=diffs**2; sc=np.full(T,np.nan)
            for t in range(w,T):
                f=np.mean(sq[t-w+1:t+1],axis=0); sc[t]=np.max(np.maximum(-np.log(f/self.fr+self.eps),0.0))
            return sc
        def score(self,y,mu,sg,raw=None):
            if raw is None: raw=y
            d=(np.mean(((y-mu)/sg)**2,axis=1)-self.d_mn)/self.d_sd
            s=self._stat(raw); s=np.nan_to_num((s-self.s_mn)/self.s_sd,nan=0.0)
            return np.maximum(d,s)

    conf_sc=ConformityScorer(10); conf_sc.fit(val_y,val_mu,val_sg)
    var_sc=VarianceScorer(10); var_sc.fit(val_y,val_mu,val_sg)
    fde_sc=FDEScorer(5); fde_sc.fit(val_y,val_mu,val_sg)

    #Fit TranAD normalisation
    tr_mu=tr_sd=None
    if tranad_model is not None:
        tr_errs=[]
        for eid in sorted(splits["val"]["unit_nr"].unique())[:5]:
            vals=splits["val"][splits["val"]["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
            ty,p1,p2=_infer_tranad(tranad_model,vals,W,device)
            if ty is not None: tr_errs.extend(np.mean((ty-p2)**2,axis=1).tolist())
        if tr_errs: tr_mu=float(np.mean(tr_errs)); tr_sd=max(float(np.std(tr_errs)),1e-8)

    #Collect scores per type per method
    methods = list(METHOD_LINE.keys())
    all_scores = {m:{at:{"s":[],"l":[]} for at in ANOM_TYPES} for m in methods}

    for eid in eids:
        ed=_ed(splits["test"],eid,sensors); T=len(ed["sensor_values"])
        if T<=W+20: continue
        for at in ANOM_TYPES:
            try:
                traj=gen.create_injected_trajectory(ed,at,injection_life_frac=0.5,magnitude=4.0,duration=15)
                y,mu,sg=_infer_gru(model,traj.sensor_values,W,device)
                if y is None: continue
                n=len(y); lab=traj.labels[W:W+n]; raw=traj.sensor_values[W:W+n]
                nll_s,_=nll_sc.score(y,mu,sg,normalize=True)
                d_conf=conf_sc.score(y,mu,sg,raw)
                d_var=var_sc.score(y,mu,sg,raw)
                d_fde=fde_sc.score(y,mu,sg,raw)
                urd_s=urd.score(y,mu,sg)["combined"]
                for m,s in [("NLL",nll_s[:n]),("D+Conformity",d_conf[:n]),("D+Variance",d_var[:n]),("D+FDE",d_fde[:n]),("URD (baseline)",urd_s[:n])]:
                    all_scores[m][at]["s"].extend(s.tolist()); all_scores[m][at]["l"].extend(lab.tolist())
                if tranad_model is not None:
                    ty,p1,p2=_infer_tranad(tranad_model,traj.sensor_values,W,device)
                    if ty is not None:
                        ts=_tranad_score(ty[:n],p1[:n],p2[:n],tr_mu,tr_sd)
                        all_scores["TranAD"][at]["s"].extend(ts.tolist()); all_scores["TranAD"][at]["l"].extend(lab.tolist())
                from sklearn.ensemble import IsolationForest
                wins=np.stack([traj.sensor_values[i:i+W] for i in range(n)],axis=0)
                if_s=-IsolationForest(contamination=0.05,random_state=42,n_jobs=-1).fit(wins.reshape(n,-1)).score_samples(wins.reshape(n,-1))
                all_scores["IForest"][at]["s"].extend(if_s.tolist()); all_scores["IForest"][at]["l"].extend(lab.tolist())
            except Exception:
                continue

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    flat = axes.flatten()
    table_rows = []
    for idx, at in enumerate(ANOM_TYPES):
        ax = flat[idx]
        row = [at.replace("_"," ").title()]
        for mn in methods:
            s=np.array(all_scores[mn][at]["s"]); l=np.array(all_scores[mn][at]["l"])
            if len(s)==0 or len(np.unique(l))<2: row.append("n/a"); continue
            try:
                auc_v=roc_auc_score(l,s); fpr,tpr,_=roc_curve(l,s)
                kw=METHOD_LINE[mn]
                ax.plot(fpr,tpr,label=f"{mn} ({auc_v:.3f})",alpha=0.92,**kw)
            except Exception: auc_v=float("nan")
            row.append(f"{auc_v:.3f}" if not (isinstance(auc_v,float) and np.isnan(auc_v)) else "n/a")
        table_rows.append(row)
        ax.plot([0,1],[0,1],"k--",lw=0.7,alpha=0.2)
        ax.set_xlim([0,1]); ax.set_ylim([0,1.02])
        ax.set_xlabel("False Positive Rate",fontsize=9); ax.set_ylabel("True Positive Rate",fontsize=9)
        title_col=PAL["blue"] if at=="sensor_freeze" else "#1A237E"
        ax.set_title(at.replace("_"," ").title(),fontweight="bold",color=title_col,fontsize=10.5)
        ax.legend(fontsize=6.5,loc="lower right")

    flat[5].axis("off")
    flat[5].text(0.5,0.55,
        f"KEY RESULT\n\nSensor Freeze:\n"
        f"  NLL    = {FREEZE_ROC['NLL']:.4f}  (sub-random)\n"
        f"  TranAD = {FREEZE_ROC['TranAD']:.4f}  (residual-only)\n"
        f"  URD    = {FREEZE_ROC['URD (baseline)']:.4f}  (+{FREEZE_ROC['URD (baseline)']-FREEZE_ROC['NLL']:.4f})\n\n"
        "URD baseline uses\ncalibrated Mahalanobis D\n+ tuned FDE/run S\n+ weighted fusion\n= comprehensive detection",
        transform=flat[5].transAxes,ha="center",va="center",fontsize=10.5,fontweight="bold",color=PAL["blue"],
        bbox=dict(boxstyle="round,pad=0.65",facecolor="#E3F2FD",alpha=0.97,edgecolor=PAL["blue"],lw=1.5))
    fig.suptitle("ROC Curves by Anomaly Type — All 7 Methods\n"
                 "Bold solid = URD baseline  |  Dashed-dot = TranAD  |  Others = ablation path",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig4_roc_curves_by_type.png")
    _csv_write(table_rows, ["Anomaly Type"]+methods, "table1_anomaly_detection.csv")


#═══════════════════════════════════════════════════════════════════════════════
#FIG 5 — Case-study timeline with freeze
#═══════════════════════════════════════════════════════════════════════════════
def fig5_case_timeline_freeze(model, urd, test_df, sensors, W, device="cpu"):
    _style()
    from src.synthetic.anomaly_generator import AnomalyGenerator
    gen = AnomalyGenerator(sensors, random_seed=42)
    for eid in sorted(test_df["unit_nr"].unique()):
        ed = _ed(test_df, eid, sensors)
        if len(ed["sensor_values"]) < 80: continue
        traj = gen.create_injected_trajectory(ed, "sensor_freeze", injection_life_frac=0.5, magnitude=4.0, duration=20)
        vals = traj.sensor_values; labels = traj.labels; lf = traj.life_fracs
        y,mu,sg = _infer_gru(model, vals, W, device)
        if y is None: continue
        n=len(y); lf_p=lf[W:W+n]
        res = urd.score(y, mu, sg)
        idx_s = sensors.index("s_4") if "s_4" in sensors else 0

        fig, axes = plt.subplots(3,1,figsize=(12,9),sharex=True)
        #Panel 1: raw signal + GRU prediction
        axes[0].plot(lf_p, y[:,idx_s], color=PAL["red"], lw=0.95, alpha=0.85, label=f"True {sensors[idx_s]}")
        axes[0].plot(lf_p, mu[:,idx_s], color=PAL["blue"], lw=1.3, label="GRU pred μ")
        axes[0].fill_between(lf_p, mu[:,idx_s]-2*sg[:,idx_s], mu[:,idx_s]+2*sg[:,idx_s], color=PAL["blue"], alpha=0.13, label="μ ± 2σ")
        axes[0].set_ylabel(f"Sensor {sensors[idx_s]}", fontweight="bold")
        axes[0].legend(loc="upper left", fontsize=9)
        axes[0].set_title(f"GRU Prediction Band — Sensor Freeze Case Study  (Engine #{eid})", fontweight="bold")

        #Panel 2: combined URD score + label
        axes[1].plot(lf_p, np.clip(res["combined"][:n],0,None), color=PAL["purple"], lw=1.8, label="URD combined score")
        axes[1].fill_between(lf_p, 0, np.clip(res["combined"][:n],0,None), alpha=0.13, color=PAL["purple"])
        axes[1].plot(lf_p, (labels[W:W+n]>0).astype(int)*res["combined"][:n].max(), color=PAL["gray"], lw=0.9, alpha=0.5, ls="--", label="Anomaly window (scaled)")
        axes[1].set_ylabel(r"$A_t=0.35\tilde{D}+0.65\tilde{S}$", fontweight="bold")
        axes[1].legend(loc="upper left", fontsize=9)

        #Panel 3: individual D/U/S channels
        axes[2].plot(lf_p, np.clip(res["deviation"][:n],0,None), color=PAL["red"], lw=1.6, label="D (Mahalanobis)")
        axes[2].plot(lf_p, res["uncertainty"][:n], color=PAL["green"], lw=1.6, label="U (sigma inflation)")
        axes[2].plot(lf_p, np.clip(res["stationarity"][:n],0,None), color=PAL["blue"], lw=1.6, label="S (FDE+run)")
        axes[2].axhline(1.0, color=PAL["gray"], ls="--", lw=0.8, alpha=0.5)
        axes[2].set_xlabel("Life Fraction", fontweight="bold")
        axes[2].set_ylabel("Channel Value", fontweight="bold")
        axes[2].legend(loc="upper left", fontsize=9)
        axes[2].set_title(r"D = calibrated Mahalanobis  |  U = sigma inflation  |  S = tuned FDE+run")

        fig.suptitle("Case Study — Injected Sensor Freeze: GRU Predictions and URD Channel Behaviour\n"
                     "Note: S channel spikes while D stays low — URD baseline catches what residual-only methods miss",
                     fontsize=12, fontweight="bold")
        plt.tight_layout(); _save(fig, "fig5_case_timeline_freeze.png"); break


#═══════════════════════════════════════════════════════════════════════════════
#FIG 6 — D/U/S distributions by category (restyled, fixed boxplot)
#═══════════════════════════════════════════════════════════════════════════════
def fig6_dus_distributions(model, urd, test_df, sensors, W, device="cpu"):
    from src.synthetic.anomaly_generator import AnomalyGenerator
    _style()
    gen = AnomalyGenerator(sensors, random_seed=42)
    cats = sorted(set(CAT_MAP.values()))
    vals = {c:{"D":[],"U":[],"S":[]} for c in cats}

    for eid in sorted(test_df["unit_nr"].unique())[:12]:
        ed = _ed(test_df, eid, sensors); base=ed["sensor_values"]; T=len(base)
        if T <= W+40: continue
        for at in ANOM_TYPES:
            try:
                traj=gen.create_injected_trajectory(ed, at, 0.5, 4.0, 15)
                y,mu,sg=_infer_gru(model, traj.sensor_values, W, device)
                if y is None: continue
                res=urd.score(y, mu, sg); mask=(traj.labels[W:W+len(y)]>0)
                if mask.sum()==0: continue
                cat=CAT_MAP[at]
                vals[cat]["D"].append(float(np.mean(np.clip(res["deviation"][mask],0,None))))
                vals[cat]["U"].append(float(np.mean(res["uncertainty"][mask])))
                vals[cat]["S"].append(float(np.mean(np.clip(res["stationarity"][mask],0,None))))
            except Exception: continue
        for dt in DRIFT_TYPES:
            try:
                dv,ds=_inject_drift(base, dt, T); y,mu,sg=_infer_gru(model, dv, W, device)
                if y is None: continue
                res=urd.score(y, mu, sg)
                es=max(0,ds-W); mask=np.zeros(len(y),dtype=bool); mask[es:]=True
                if mask.sum()==0: continue
                vals["drift"]["D"].append(float(np.mean(np.clip(res["deviation"][mask],0,None))))
                vals["drift"]["U"].append(float(np.mean(res["uncertainty"][mask])))
                vals["drift"]["S"].append(float(np.mean(np.clip(res["stationarity"][mask],0,None))))
            except Exception: continue

    fig, axes = plt.subplots(1,3,figsize=(16,5.8),sharey=False)
    ch_colors = [PAL["red"], PAL["green"], PAL["blue"]]
    ch_labels = ["D (Mahalanobis deviation)","U (sigma inflation)","S (tuned FDE+run)"]
    short_cats = [c.replace("_"," ").replace("point anomaly","point\nanom.").replace("persistent shift","persist.\nshift").replace("noise anomaly","noise\nanom.").replace("sensor malfunction","sensor\nmalfn.") for c in cats]

    for ax, ch, col, clabel in zip(axes, ["D","U","S"], ch_colors, ch_labels):
        data = [vals[c][ch] if vals[c][ch] else [0.0] for c in cats]
        bp = ax.boxplot(data, tick_labels=short_cats, showfliers=False, patch_artist=True, medianprops=dict(color="white",lw=2))
        for patch in bp["boxes"]: patch.set_facecolor(col); patch.set_alpha(0.55)
        for whisker in bp["whiskers"]: whisker.set(color=col, alpha=0.7, lw=1.2)
        for cap in bp["caps"]: cap.set(color=col, alpha=0.7, lw=1.2)
        ax.set_title(clabel, fontweight="bold", fontsize=11, pad=8, color=col)
        ax.set_xlabel("Anomaly Category", fontweight="bold")
        ax.tick_params(axis="x", labelsize=9)
        if ax == axes[0]: ax.set_ylabel("Mean Channel Value in Event Window", fontweight="bold")

    fig.suptitle("URD Channel Value Distributions by Anomaly Category\n"
                 r"D = calibrated Mahalanobis  |  U = sigma inflation  |  S = tuned FDE+run  |  $A_t=0.35\tilde{D}+0.65\tilde{S}$",
                 fontsize=12.5, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig6_dus_distributions.png")


#─── helper reused by fig7 and fig10 ─────────────────────────────────────────
def _draw_importance_bars(stage_d, model_key, acc_str, title_line1, out_name):
    """Shared renderer for feature-importance bar charts (fig7 and fig10)."""
    _style()
    if not stage_d:
        print(f"  skipped {out_name} — no Stage D results"); return
    key = model_key
    if key not in stage_d:
        for k in stage_d:
            if model_key.split("_")[0] in k and "URD" in k: key=k; break
    if key not in stage_d or "feature_importance" not in stage_d[key]:
        print(f"  skipped {out_name} — no feature_importance in {key}"); return
    imp = stage_d[key]["feature_importance"]
    items = sorted(imp.items(), key=lambda kv: kv[1])
    names = [k for k,_ in items]; vals = [v for _,v in items]
    URD_FEATS = {"deviation_at_peak","uncertainty_at_peak","stationarity_at_peak","uncertainty_slope","stationarity_max","du_ratio","signed_deviation_mean"}
    colors = [PAL["blue"] if n in URD_FEATS else PAL["gray"] for n in names]
    fig, ax = plt.subplots(figsize=(10.5, 7.5))
    bars = ax.barh(range(len(names)), vals, color=colors, edgecolor="white", height=0.72)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=10.5)
    ax.set_xlabel("Feature Importance (Gini impurity reduction)", fontweight="bold", fontsize=11)
    ax.set_title(f"{title_line1}\n{acc_str}", fontweight="bold", fontsize=12)
    mx = max(vals) if vals else 1.0
    for bar,v in zip(bars,vals):
        ax.text(bar.get_width()+mx*0.012, bar.get_y()+bar.get_height()/2, f"{v:.3f}", va="center", fontsize=8.5, color="#37474F")
    patches = [mpatches.Patch(color=PAL["blue"],label="URD channel features (D, U, S)"),
               mpatches.Patch(color=PAL["gray"],label="Standard score-shape features")]
    ax.legend(handles=patches, loc="lower right", fontsize=10.5)
    plt.tight_layout(); _save(fig, out_name)


#═══════════════════════════════════════════════════════════════════════════════
#FIG 7 — Feature importance (RF best model) and FIG 10 — XGBoost importance
#Both use _draw_importance_bars — identical design, different model key
#═══════════════════════════════════════════════════════════════════════════════
def fig7_feature_importance(stage_d):
    _draw_importance_bars(stage_d, "random_forest_URD_16feat", "Best model: RF + 16-feature URD  |  Accuracy = 0.955  |  Drift→Anomaly = 2.9%", "Random Forest Feature Importances — Drift vs Anomaly Classification", "fig7_feature_importance.png")


def fig10_xgboost_importance(stage_d):
    _draw_importance_bars(stage_d, "xgboost_URD_16feat", "XGBoost + 16-feature URD  |  Accuracy = 0.951  |  Drift→Anomaly = 4.4%", "XGBoost Feature Importances — Drift vs Anomaly Classification", "fig10_stage_d_feature_importance.png")


#═══════════════════════════════════════════════════════════════════════════════
#FIG 8 — Probabilistic calibration (fixed math.erf bug)
#═══════════════════════════════════════════════════════════════════════════════
def fig8_probabilistic_calibration(model, urd, val_df, sensors, W, device="cpu"):
    _style()
    rs = []
    for eid in sorted(val_df["unit_nr"].unique()):
        vals = val_df[val_df["unit_nr"]==eid].sort_values("time_cycles")[sensors].values
        y,mu,sg = _infer_gru(model, vals, W, device)
        if y is None: continue
        #Use calibrated sigma if available (sigma_temp stored on URD scorer)
        if hasattr(urd,"sigma_temp") and urd.sigma_temp is not None:
            sg_eff = sg * urd.sigma_temp[np.newaxis,:]
        else:
            sg_eff = sg
        rs.append(((y - mu) / np.maximum(sg_eff, 1e-8)).reshape(-1))
    if not rs: print("  skipped fig8 — no validation data"); return
    residuals = np.concatenate(rs)

    z_grid = np.array([0.5,1.0,1.5,2.0,2.5,3.0])
    empirical = [float(np.mean(np.abs(residuals) <= z)) for z in z_grid]
    theoretical = [float(math.erf(z / math.sqrt(2.0))) for z in z_grid]  #Fixed: use math.erf not np.math.erf

    fig, axes = plt.subplots(1,2,figsize=(13,5.5))
    axes[0].plot(z_grid, empirical, marker="o", color=PAL["blue"], lw=1.8, label="Empirical coverage")
    axes[0].plot(z_grid, theoretical, marker="s", color=PAL["gray"], lw=1.5, ls="--", label="Gaussian ideal")
    axes[0].set_xlabel(r"$z$ in $\mu \pm z\sigma$", fontweight="bold")
    axes[0].set_ylabel("Coverage probability", fontweight="bold")
    axes[0].set_title("Coverage Calibration — Calibrated GRU\n(after sigma-temperature scaling)", fontweight="bold")
    axes[0].legend(fontsize=10)

    axes[1].hist(residuals, bins=50, density=True, color=PAL["blue"], alpha=0.65, edgecolor="white")
    x = np.linspace(-4.5, 4.5, 500)
    pdf = (1/math.sqrt(2*math.pi)) * np.exp(-0.5*x**2)
    axes[1].plot(x, pdf, color=PAL["red"], lw=1.8, label="N(0,1)")
    axes[1].set_xlabel(r"Normalised residual $r_{t,j}=(x_{t,j}-\mu_{t,j})/\sigma^{eff}_{t,j}$", fontweight="bold")
    axes[1].set_ylabel("Density", fontweight="bold")
    axes[1].set_title(f"Residual Distribution\nmean={residuals.mean():.3f}  std={residuals.std():.3f}  (ideal: 0 and 1)", fontweight="bold")
    axes[1].legend(fontsize=10)

    fig.suptitle("Probabilistic Calibration of the URD Backbone\n"
                 r"Well-calibrated $\sigma$ is required for valid Mahalanobis D, meaningful U channel, and reliable S normalisation",
                 fontsize=12, fontweight="bold")
    plt.tight_layout(); _save(fig, "fig8_probabilistic_calibration.png")


#═══════════════════════════════════════════════════════════════════════════════
#TABLES
#═══════════════════════════════════════════════════════════════════════════════
def table1_anomaly_detection(stage_c):
    if not stage_c: return
    rows=[]; all_types=sorted(stage_c.get("urd",{}).get("per_type",{}).keys())
    for t in all_types:
        rows.append([t,
            stage_c["urd"]["per_type"][t].get("roc_auc",""),
            stage_c["urd"]["per_type"][t].get("pr_auc",""),
            stage_c.get("tranad",{}).get("per_type",{}).get(t,{}).get("roc_auc",""),
            stage_c.get("tranad",{}).get("per_type",{}).get(t,{}).get("pr_auc","")])
    rows.append(["OVERALL",stage_c["urd"]["overall"]["roc_auc"],stage_c["urd"]["overall"]["pr_auc"],stage_c.get("tranad",{}).get("overall",{}).get("roc_auc",""),stage_c.get("tranad",{}).get("overall",{}).get("pr_auc","")])
    _csv_write(rows,["type","urd_roc","urd_pr","tranad_roc","tranad_pr"],"table1_anomaly_detection.csv")


def table2_drift_ablation(stage_d):
    if not stage_d: return
    rows=[]
    for name,data in sorted(stage_d.items()):
        rows.append([name,data.get("accuracy",""),data.get("drift_as_anomaly_rate",""),data.get("anomaly_as_drift_rate","")])
    _csv_write(rows,["model","accuracy","drift_as_anomaly_rate","anomaly_as_drift_rate"],"table2_drift_ablation.csv")


def table3_fingerprint(stage_e):
    if not stage_e: return
    rows=[]; rep=stage_e.get("five_class",{}).get("report",{})
    for cls in ["drift","noise_anomaly","persistent_shift","point_anomaly","sensor_malfunction"]:
        if cls in rep:
            rows.append([cls,rep[cls].get("precision",""),rep[cls].get("recall",""),rep[cls].get("f1-score",""),rep[cls].get("support","")])
    rows.append(["ACCURACY",stage_e.get("five_class",{}).get("accuracy",""),"","",""])
    _csv_write(rows,["class","precision","recall","f1","support"],"table3_fingerprint.csv")


def table4_model_comparison(stage_a):
    if not stage_a: return
    rows=[]
    for k,res in stage_a.get("eval_results",{}).items():
        rows.append([k,res.get("mse",""),res.get("mae",""),res.get("mean_sigma","")])
    _csv_write(rows,["model","test_mse","test_mae","mean_sigma"],"table4_model_comparison.csv")


def copy_optional_figure(src_name, dst_name):
    src = os.path.join(FIG, src_name)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(PAPER, dst_name))
        print(f"  copied {dst_name}")


#═══════════════════════════════════════════════════════════════════════════════
#MAIN
#═══════════════════════════════════════════════════════════════════════════════
def main():
    _style()
    cfg = _cfg()
    W = cfg["preprocessing"]["window_size"]

    #Load all stage result JSONs (graceful if missing)
    stage_c = _load_json(os.path.join(RES,"stage_c_results.json"))
    stage_d = _load_json(os.path.join(RES,"stage_d_results.json"))
    stage_e = _load_json(os.path.join(RES,"stage_e_results.json"))
    stage_a = _load_json(os.path.join(RES,"stage_a_results.json"))

    print("="*65); print("  PAPER OUTPUT GENERATOR  →  outputs/for_paper/"); print("="*65)
    print(f"\n[Phase 1 — no data needed]")
    fig1_pipeline_overview()

    print(f"\n[Phase 2 — loading data and model checkpoints]")
    try:
        splits, sensors = _load_data(cfg)
    except Exception as e:
        print(f"  ERROR loading data: {e}"); return
    try:
        model = _load_gru(cfg)
    except FileNotFoundError as e:
        print(f"  ERROR — run 01_train_baselines first: {e}"); return
    tranad_model = _load_tranad(cfg)
    if tranad_model is None: print("  Note: TranAD checkpoint not found — TranAD panels will be skipped")

    print(f"[Phase 3 — fitting scorers on validation data]")
    urd = _fit_urd(model, splits["val"], sensors, W)
    nll_sc = _fit_nll(model, splits["val"], sensors, W)

    print(f"[Phase 4 — generating figures and tables]")
    fig2_urd_channels(model, urd, splits["test"], sensors, W)
    fig3_sensor_freeze_blind_spot(model, urd, nll_sc, splits["test"], sensors, W, tranad_model)
    fig5_case_timeline_freeze(model, urd, splits["test"], sensors, W)
    fig6_dus_distributions(model, urd, splits["test"], sensors, W)
    fig8_probabilistic_calibration(model, urd, splits["val"], sensors, W)

    print(f"[Phase 5 — figures from Stage C results]")
    fig_roc_pr_urd_vs_tranad(stage_c)
    fig_threshold_sweep()
    fig_per_type_pr(stage_c)
    fig4_roc_curves_by_type(cfg, splits, sensors, W, model, urd, nll_sc, tranad_model)

    print(f"[Phase 6 — Stage D / E figures]")
    fig7_feature_importance(stage_d)
    fig10_xgboost_importance(stage_d)

    print(f"[Phase 7 — tables]")
    table1_anomaly_detection(stage_c)
    table2_drift_ablation(stage_d)
    table3_fingerprint(stage_e)
    table4_model_comparison(stage_a)

    copy_optional_figure("fingerprint_5class_cm.png","fig9_fingerprint_5class_confusion.png")
    copy_optional_figure("fingerprint_9class_cm.png","fig10_fingerprint_9class_confusion.png")

    print("\n"+"="*65); print(f"  Done.  All outputs → {PAPER}"); print("="*65)


if __name__=="__main__":
    main()