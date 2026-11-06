import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# ------------------------------------------------------------------
# Page config
# ------------------------------------------------------------------
st.set_page_config(page_title="AI Network Intrusion Detection", page_icon="🛡️",
                   layout="wide", initial_sidebar_state="collapsed")

BL, CY, RED, GRN, AMB = "#3b82f6", "#38bdf8", "#ff6b81", "#34d399", "#fbbf24"
TXT, MUT = "#eaf3ff", "#9db8dc"
COLORS = {"BENIGN": BL, "ATTACK": RED}

# ------------------------------------------------------------------
# Look and feel: dark blue glass dashboard, compact so it fits one screen
# ------------------------------------------------------------------
st.markdown("""
<style>
.stApp {
  background:
    radial-gradient(circle at 88% 0%, rgba(24,100,200,0.55) 0%, rgba(24,100,200,0) 38%),
    radial-gradient(circle at 4% 100%, rgba(20,80,170,0.45) 0%, rgba(20,80,170,0) 40%),
    linear-gradient(160deg, #030a1f 0%, #06214d 60%, #071b3d 100%);
  color: #eaf3ff;
}
header[data-testid="stHeader"], footer, #MainMenu {visibility: hidden; height: 0;}
section[data-testid="stSidebar"], div[data-testid="collapsedControl"] {display: none;}
.block-container {padding-top: 0.8rem; padding-bottom: 0.3rem; max-width: 100%;}
div[data-testid="stVerticalBlock"] {gap: 0.45rem;}
div[data-testid="stVerticalBlockBorderWrapper"] {
  background: rgba(255,255,255,0.05);
  border: 1px solid rgba(125,190,255,0.30) !important;
  border-radius: 14px;
  padding: 0.15rem 0.35rem;
}
.main-title {font-size: 1.85rem; font-weight: 800; color: #fff; line-height: 1.15;}
.subtitle {color: #38bdf8; font-size: 0.92rem; margin-bottom: 0.3rem;}
.ptitle {font-weight: 700; font-size: 0.8rem; letter-spacing: 0.04em; color: #eaf3ff; margin: 0.1rem 0 0.1rem 0;}
.sub {font-size: 0.78rem; color: #38bdf8; font-weight: 700; text-align: center;}
.chips {display: flex; gap: 8px; flex-wrap: wrap;}
.chip {background: rgba(255,255,255,0.07); border: 1px solid rgba(125,190,255,0.30);
       border-radius: 10px; padding: 3px 10px; font-size: 0.74rem; color: #9db8dc; line-height: 1.25;}
.chip b {display: block; font-size: 1.0rem; color: #fff;}
.verdict {font-size: 1.9rem; font-weight: 800; line-height: 1.1; margin: 0.2rem 0;}
label[data-testid="stWidgetLabel"] p {font-size: 0.76rem; color: #9db8dc;}
div[data-testid="stRadio"] label p {font-size: 0.78rem;}
button[data-baseweb="tab"] {background: rgba(255,255,255,0.05); border-radius: 10px 10px 0 0; padding: 4px 14px;}
button[data-baseweb="tab"][aria-selected="true"] {background: rgba(56,189,248,0.20);}
div[data-baseweb="tab-list"] {gap: 6px;}
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------
# Load model, features, sample data
# ------------------------------------------------------------------
@st.cache_resource
def load_model():
    return joblib.load("models/rf_compressed.joblib")


@st.cache_data
def load_features():
    return joblib.load("models/features.joblib")


@st.cache_data
def load_sample():
    df = pd.read_csv("data/sample_test.csv")
    df["Class"] = df["Label"].map({0: "BENIGN", 1: "ATTACK"})
    return df


model = load_model()
features = load_features()
sample = load_sample()

imp_series = pd.Series(model.feature_importances_, index=features).sort_values(ascending=False)
benign = sample[sample["Class"] == "BENIGN"][features]
b_mean = benign.mean()
b_std = benign.std().replace(0, 1).fillna(1)


@st.cache_data
def sample_scores():
    return model.predict_proba(sample[features])[:, 1]


# Class counts: before cleaning and after cleaning (duplicates removed)
CLS = pd.DataFrame({
    "Label": ["BENIGN", "DoS Hulk", "PortScan", "DDoS", "DoS GoldenEye", "FTP-Patator", "SSH-Patator",
              "DoS slowloris", "DoS Slowhttptest", "Bot", "Web Brute Force", "Web XSS",
              "Infiltration", "Web SQL Injection", "Heartbleed"],
    "Before": [2273097, 231073, 158930, 128027, 10293, 7938, 5897, 5796, 5499, 1966, 1507, 652, 36, 21, 11],
    "After": [2072418, 172846, 90694, 128014, 10286, 5931, 3219, 5385, 5228, 1948, 1470, 652, 36, 21, 11],
})

RESULTS = pd.DataFrame({
    "Metric": ["Accuracy", "Precision", "Recall", "F1", "Macro F1"],
    "Decision Tree": [0.9984, 0.9959, 0.9949, 0.9954, 0.9972],
    "Random Forest": [0.9986, 0.9964, 0.9952, 0.9958, 0.9974]})
CMS = {"Decision Tree": np.array([[414135, 349], [433, 84715]]),
       "Random Forest": np.array([[414175, 309], [412, 84736]])}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def ptitle(text):
    st.markdown(f'<div class="ptitle">{text}</div>', unsafe_allow_html=True)


def chips(items):
    html = '<div class="chips">' + "".join(
        f'<div class="chip">{lbl}<b style="color:{col}">{val}</b></div>' for lbl, val, col in items) + "</div>"
    st.markdown(html, unsafe_allow_html=True)


def style(fig, h, legend):
    fig.update_layout(height=h, margin=dict(l=6, r=6, t=6, b=6),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                      font=dict(color=TXT, size=11))
    if legend == "top":
        fig.update_layout(showlegend=True, legend=dict(orientation="h", y=1.08, x=1, xanchor="right",
                                                       bgcolor="rgba(0,0,0,0)", title_text=""))
    elif legend is False:
        fig.update_layout(showlegend=False)
    fig.update_xaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    fig.update_yaxes(gridcolor="rgba(255,255,255,0.08)", zeroline=False)
    return fig


def show(fig, h, legend=False, key=None):
    st.plotly_chart(style(fig, h, legend), use_container_width=True,
                    config={"displayModeBar": False}, theme=None, key=key)


def counts(s, y, t):
    p = s >= t
    tp = int((p & (y == 1)).sum())
    fp = int((p & (y == 0)).sum())
    fn = int((~p & (y == 1)).sum())
    tn = int((~p & (y == 0)).sum())
    return tp, fp, fn, tn


def heat(z, text_fmt, h, key):
    fig = go.Figure(go.Heatmap(z=z, x=["Pred BENIGN", "Pred ATTACK"], y=["Actual BENIGN", "Actual ATTACK"],
                               colorscale="Blues", showscale=False, texttemplate=text_fmt,
                               textfont=dict(size=12, color="white")))
    fig.update_yaxes(autorange="reversed", tickfont_size=9)
    fig.update_xaxes(tickfont_size=9)
    show(fig, h, False, key)


# ------------------------------------------------------------------
# Header (global): title, quick numbers, screen-fit control
# ------------------------------------------------------------------
hl, hr = st.columns([3.4, 1.3])
with hl:
    st.markdown('<div class="main-title">Network Intrusion Detection with Machine Learning</div>'
                '<div class="subtitle">CIC-IDS-2017  |  3.1M network flows  |  '
                'Data cleaning  &rarr;  Decision Tree vs Random Forest</div>', unsafe_allow_html=True)
    chips([("Clean flows", "2,498,159", CY), ("Features", "71", CY),
           ("RF F1 (attack)", "0.9958", GRN), ("Missed attacks (RF)", "412", RED)])
with hr:
    size = st.select_slider("Screen fit (chart size). If you still need to scroll, choose Small.",
                            ["Small", "Medium", "Large"], value="Medium", key="size")
H = {"Small": 185, "Medium": 225, "Large": 280}[size]

tabs = st.tabs(["📊 Overview", "🔬 Feature Explorer", "🎚️ Threshold Lab", "🧪 New Prediction", "⚡ Live Flow", "📦 Batch Scan"])


# ==================================================================
# TAB 1: OVERVIEW (same layout as the project cover image)
# ==================================================================
def tab_overview():
    c1, c2, c3 = st.columns(3)

    with c1:
        with st.container(border=True):
            ptitle("1. BEFORE vs AFTER CLEANING: classes")
            a, b = st.columns([3.2, 1])
            mode = a.radio("View", ["Before", "After", "Compare"], horizontal=True,
                           key="ov_mode", label_visibility="collapsed")
            log = b.toggle("Log", True, key="ov_log")
            d = CLS.sort_values("Before")
            fig = go.Figure()
            if mode == "Compare":
                fig.add_trace(go.Bar(y=d.Label, x=d.Before, orientation="h", name="Before", marker_color=AMB))
                fig.add_trace(go.Bar(y=d.Label, x=d.After, orientation="h", name="After", marker_color=GRN))
                fig.update_layout(barmode="group")
                legend = "top"
            else:
                vals = d.Before if mode == "Before" else d.After
                cols = [BL if lbl == "BENIGN" else RED for lbl in d.Label]
                fig.add_trace(go.Bar(y=d.Label, x=vals, orientation="h", marker_color=cols,
                                     hovertemplate="%{y}: %{x:,}<extra></extra>"))
                legend = False
            fig.update_xaxes(type="log" if log else "linear")
            fig.update_yaxes(dtick=1, tickfont_size=9)
            show(fig, H, legend, "ov1")

    with c2:
        with st.container(border=True):
            ptitle("2. CLEANING PIPELINE")
            what = st.radio("Metric", ["Rows", "Columns"], horizontal=True,
                            key="ov_pipe", label_visibility="collapsed")
            if what == "Rows":
                names = ["Raw (8 CSV)", "NaN/Inf removed", "Duplicates removed"]
                vals = [3119345, 2827876, 2498159]
            else:
                names = ["Raw", "IDs dropped", "Constants dropped"]
                vals = [85, 80, 72]
            fig = go.Figure(go.Bar(x=names, y=vals, marker_color=[BL, CY, GRN],
                                   text=[f"{v:,}" for v in vals], textposition="outside", cliponaxis=False))
            fig.update_yaxes(range=[0, max(vals) * 1.2], showticklabels=False)
            show(fig, H, False, "ov2")

    with c3:
        with st.container(border=True):
            ptitle("3. AFTER CLEANING: labels")
            view = st.radio("Labels", ["Binary", "Attack types"], horizontal=True,
                            key="ov_lab", label_visibility="collapsed")
            if view == "Binary":
                fig = go.Figure(go.Pie(labels=["BENIGN", "ATTACK"], values=[2072418, 425741], hole=0.58,
                                       marker=dict(colors=[BL, RED]), textinfo="label+percent",
                                       sort=False))
                fig.add_annotation(text="2,498,159<br>clean flows", showarrow=False, font=dict(size=13))
                show(fig, H, False, "ov3a")
            else:
                att = CLS[CLS.Label != "BENIGN"]
                fig = go.Figure(go.Pie(labels=att.Label, values=att.After, hole=0.58, textinfo="percent",
                                       marker=dict(colors=px.colors.qualitative.Vivid)))
                fig.update_layout(showlegend=True, legend=dict(orientation="v", x=1.0, y=0.5, font_size=9))
                show(fig, H, "keep", "ov3b")

    d1, d2 = st.columns([1.15, 1])

    with d1:
        with st.container(border=True):
            ptitle("4. MODEL COMPARISON (test set)")
            pick = st.radio("Models", ["Both", "Decision Tree", "Random Forest"], horizontal=True,
                            key="ov_models", label_visibility="collapsed")
            models = ["Decision Tree", "Random Forest"] if pick == "Both" else [pick]
            fig = go.Figure()
            for mname, col in zip(["Decision Tree", "Random Forest"], [CY, GRN]):
                if mname in models:
                    fig.add_trace(go.Bar(x=RESULTS["Metric"], y=RESULTS[mname], name=mname, marker_color=col,
                                         text=[f"{v:.4f}" for v in RESULTS[mname]], textposition="outside",
                                         textfont=dict(size=9), cliponaxis=False))
            fig.update_layout(barmode="group")
            fig.update_yaxes(range=[0.990, 1.0014], tickformat=".3f")
            show(fig, H, "top", "ov4")

    with d2:
        with st.container(border=True):
            ptitle("5. ERRORS: confusion matrices")
            pct = st.toggle("Show as % of each actual class", False, key="ov_pct")
            m1, m2 = st.columns(2)
            for col, mname in zip([m1, m2], ["Decision Tree", "Random Forest"]):
                with col:
                    st.markdown(f'<div class="sub">{mname}</div>', unsafe_allow_html=True)
                    cm = CMS[mname]
                    if pct:
                        heat(cm / cm.sum(axis=1, keepdims=True) * 100, "%{z:.2f}%", H - 15, f"ov5_{mname}")
                    else:
                        heat(cm, "%{z:,}", H - 15, f"ov5_{mname}")


# ==================================================================
# TAB 2: FEATURE EXPLORER
# ==================================================================
def tab_features():
    names = list(imp_series.index)
    r1a, r1b = st.columns(2)

    with r1a:
        with st.container(border=True):
            ptitle("FEATURE DISTRIBUTION: attack vs benign")
            a, b, c = st.columns([3, 2.4, 1])
            feat = a.selectbox("Feature", names, key="fe_feat", label_visibility="collapsed")
            kind = b.radio("Type", ["Histogram", "Box", "Violin"], horizontal=True,
                           key="fe_kind", label_visibility="collapsed")
            lg = c.toggle("Log", True, key="fe_log")
            if kind == "Histogram":
                fig = px.histogram(sample, x=feat, color="Class", barmode="overlay", nbins=50,
                                   log_y=lg, color_discrete_map=COLORS, opacity=0.75)
            elif kind == "Box":
                fig = px.box(sample, x="Class", y=feat, color="Class", log_y=lg, color_discrete_map=COLORS)
            else:
                fig = px.violin(sample, x="Class", y=feat, color="Class", box=True, color_discrete_map=COLORS)
            show(fig, H + 20, "top", "fe1")

    with r1b:
        with st.container(border=True):
            ptitle("FEATURE IMPORTANCE (Random Forest)")
            n = st.slider("Top N features", 5, 25, 12, key="fe_n", label_visibility="collapsed")
            s = imp_series.head(n).sort_values()
            fig = go.Figure(go.Bar(x=s.values, y=s.index, orientation="h", marker_color=CY))
            fig.update_yaxes(dtick=1, tickfont_size=9, automargin=True)
            show(fig, H + 20, False, "fe2")

    r2a, r2b = st.columns(2)

    with r2a:
        with st.container(border=True):
            ptitle("FEATURE vs FEATURE")
            a, b = st.columns(2)
            xf = a.selectbox("X axis", names, index=0, key="fe_x", label_visibility="collapsed")
            yf = b.selectbox("Y axis", names, index=1, key="fe_y", label_visibility="collapsed")
            fig = px.scatter(sample, x=xf, y=yf, color="Class", opacity=0.55, log_x=True, log_y=True,
                             color_discrete_map=COLORS)
            fig.update_traces(marker_size=4)
            show(fig, H + 20, "top", "fe3")

    with r2b:
        with st.container(border=True):
            ptitle("CORRELATION (top features)")
            k = st.slider("Number of features", 6, 20, 12, key="fe_k", label_visibility="collapsed")
            cols = list(imp_series.head(k).index)
            corr = sample[cols].corr()
            short = [c[:16] for c in cols]
            fig = go.Figure(go.Heatmap(z=corr.values, x=short, y=short, zmin=-1, zmax=1,
                                       colorscale="RdBu_r", showscale=False))
            fig.update_yaxes(autorange="reversed", tickfont_size=8)
            fig.update_xaxes(tickfont_size=8)
            show(fig, H + 20, False, "fe4")


# ==================================================================
# TAB 3: THRESHOLD LAB
# ==================================================================
def tab_threshold():
    s = sample_scores()
    y = sample["Label"].values
    t = st.slider("Attack threshold: lower = catches more attacks but raises false alarms",
                  0.05, 0.95, 0.5, 0.05, key="th")

    ths = np.linspace(0.02, 0.98, 49)
    P = s[None, :] >= ths[:, None]
    tp = (P & (y == 1)).sum(1)
    fp = (P & (y == 0)).sum(1)
    fn = (~P & (y == 1)).sum(1)
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / np.maximum(tp + fn, 1)
    f1 = 2 * prec * rec / np.maximum(prec + rec, 1e-9)

    ctp, cfp, cfn, ctn = counts(s, y, t)
    cprec = ctp / max(ctp + cfp, 1)
    crec = ctp / max(ctp + cfn, 1)
    cf1 = 2 * cprec * crec / max(cprec + crec, 1e-9)

    a, b = st.columns([1.4, 1])
    with a:
        with st.container(border=True):
            ptitle("PRECISION / RECALL / F1 vs THRESHOLD")
            fig = go.Figure()
            for name, arr, col in [("Precision", prec, CY), ("Recall", rec, RED), ("F1", f1, GRN)]:
                fig.add_trace(go.Scatter(x=ths, y=arr, name=name, mode="lines+markers",
                                         line=dict(color=col, width=2), marker_size=4))
            fig.add_vline(x=t, line_dash="dash", line_color=AMB)
            fig.update_yaxes(range=[0.5, 1.02])
            show(fig, H + 25, "top", "th1")
    with b:
        with st.container(border=True):
            ptitle("RESULT AT THIS THRESHOLD")
            chips([("Precision", f"{cprec:.3f}", CY), ("Recall", f"{crec:.3f}", RED),
                   ("F1", f"{cf1:.3f}", GRN), ("False alarms", f"{cfp}", AMB), ("Missed", f"{cfn}", RED)])
            heat(np.array([[ctn, cfp], [cfn, ctp]]), "%{z:,}", H - 10, "th2")

    c, d = st.columns(2)
    with c:
        with st.container(border=True):
            ptitle("MODEL SCORE DISTRIBUTION")
            dfp = pd.DataFrame({"score": s, "Class": sample["Class"]})
            fig = px.histogram(dfp, x="score", color="Class", nbins=40, barmode="overlay", log_y=True,
                               color_discrete_map=COLORS, opacity=0.75)
            fig.add_vline(x=t, line_dash="dash", line_color=AMB)
            show(fig, H, "top", "th3")
    with d:
        with st.container(border=True):
            ptitle("ERRORS vs THRESHOLD: what rises and what falls")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=ths, y=fp, name="False alarms (FP)", mode="lines",
                                     line=dict(color=AMB, width=2.5)))
            fig.add_trace(go.Scatter(x=ths, y=fn, name="Missed attacks (FN)", mode="lines",
                                     line=dict(color=RED, width=2.5)))
            fig.add_vline(x=t, line_dash="dash", line_color=CY)
            show(fig, H, "top", "th4")


# ==================================================================
# TAB 4: NEW PREDICTION (enter your own data point)
# ==================================================================
def bump_nf():
    st.session_state["nf_seed"] = st.session_state.get("nf_seed", 0) + 1


def get_base(tmpl, seed):
    if tmpl == "Average benign flow":
        return b_mean.astype(float).copy()
    cls = "BENIGN" if "BENIGN" in tmpl else "ATTACK"
    pool = sample[sample["Class"] == cls]
    i = int(np.random.default_rng(seed).integers(0, len(pool)))
    return pool.iloc[i][features].astype(float).copy()


def parse_row(text):
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None, "Paste a row to see the prediction."
    first = [t.strip() for t in lines[0].split(",")]
    try:
        vals = [float(t) for t in first]
        has_header = False
    except ValueError:
        has_header = True
    if has_header:
        if len(lines) < 2:
            return None, "Header found, but there is no data row under it."
        df = pd.read_csv(io.StringIO("\n".join(lines[:2])))
        df.columns = df.columns.str.strip()
        missing = [c for c in features if c not in df.columns]
        if missing:
            return None, f"Missing columns: {missing[:5]} ..."
        return df[features].astype(float).iloc[[0]].reset_index(drop=True), None
    if len(vals) != len(features):
        return None, f"Expected {len(features)} values in the feature order, got {len(vals)}."
    return pd.DataFrame([vals], columns=features), None


def save_pred(method, tmpl, p, thr):
    hist = st.session_state.setdefault("nf_hist", [])
    hist.append({"#": len(hist) + 1, "Input": f"{method} / {tmpl}",
                 "Attack probability": round(p, 4), "Verdict": "ATTACK" if p >= thr else "BENIGN"})


def clear_hist():
    st.session_state["nf_hist"] = []


def tab_new():
    x, err = None, None
    with st.container(border=True):
        ptitle("ENTER A NEW FLOW AND PREDICT")
        a, b, c = st.columns([1.8, 1.4, 0.8])
        method = a.radio("How do you want to enter the data?",
                         ["Quick form", "Paste a row", "Edit all features"], horizontal=True, key="nf_method")
        tmpl = b.selectbox("Start from", ["Random BENIGN flow", "Random ATTACK flow", "Average benign flow"],
                           key="nf_tmpl")
        c.button("🎲 New random", on_click=bump_nf, key="nf_rand")
        seed = st.session_state.get("nf_seed", 0)
        base = get_base(tmpl, seed)
        tag = f"{tmpl}_{seed}"

        if method == "Quick form":
            keyf = list(imp_series.head(10).index)
            cols = st.columns(5)
            vals = base.copy()
            for i, f in enumerate(keyf):
                vals[f] = cols[i % 5].number_input(f[:28], value=float(base[f]), format="%.2f",
                                                   key=f"nf_{tag}_{f}")
            x = pd.DataFrame([vals[features].values.astype(float)], columns=features)
            st.caption(f"These are the 10 most important features. The other {len(features) - 10} features "
                       f"keep the values of the starting flow.")
        elif method == "Paste a row":
            l, r = st.columns([2, 1])
            txt = l.text_area("Paste one flow: either 71 comma-separated values in feature order, "
                              "or a header line followed by one data row", height=110, key=f"nf_txt_{tag}")
            with r.expander("Feature order (copy this header)"):
                st.code(",".join(features))
            x, err = parse_row(txt)
        else:
            tbl = pd.DataFrame({"Feature": features, "Value": base[features].values})
            ed = st.data_editor(tbl, hide_index=True, height=190, disabled=["Feature"],
                                use_container_width=True, key=f"nf_ed_{tag}")
            x = pd.DataFrame([ed["Value"].astype(float).values], columns=ed["Feature"].tolist())[features]

    if x is None:
        st.info(err or "Enter data to see the prediction.")
        return
    x = x.replace([np.inf, -np.inf], np.nan)
    if x.isna().any().any():
        st.error("Some values are empty or invalid. Fill every value with a number.")
        return

    p = float(model.predict_proba(x)[0, 1])

    r1, r2, r3 = st.columns([1, 1.2, 1])
    with r1:
        with st.container(border=True):
            ptitle("PREDICTION")
            thr = st.slider("Alert threshold", 0.05, 0.95, 0.5, 0.05, key="nf_thr")
            is_att = p >= thr
            v1, v2 = st.columns([1.3, 1])
            v1.markdown(f'<div class="verdict" style="color:{RED if is_att else GRN}">'
                        f'{"ATTACK" if is_att else "BENIGN"}</div>', unsafe_allow_html=True)
            v2.button("💾 Save", on_click=save_pred, args=(method, tmpl, p, thr), key="nf_save")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 30}},
                gauge={"axis": {"range": [0, 100]},
                       "steps": [{"range": [0, thr * 100], "color": "rgba(52,211,153,0.35)"},
                                 {"range": [thr * 100, 100], "color": "rgba(255,107,129,0.35)"}],
                       "threshold": {"line": {"color": AMB, "width": 3}, "value": thr * 100},
                       "bar": {"color": CY}}))
            show(fig, H - 30, False, "nf_g")

    with r2:
        with st.container(border=True):
            ptitle("WHY: HIGHER / LOWER THAN NORMAL TRAFFIC")
            z = ((x.iloc[0] - b_mean) / b_std).clip(-10, 10)
            top = z[imp_series.head(10).index].sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[RED if v > 0 else BL for v in top.values]))
            fig.update_yaxes(tickfont_size=9, automargin=True, dtick=1)
            fig.update_xaxes(title_text="red = higher than normal, blue = lower", title_font_size=9)
            show(fig, H + 25, False, "nf_d")

    with r3:
        with st.container(border=True):
            ptitle("SAVED PREDICTIONS")
            hist = st.session_state.get("nf_hist", [])
            if hist:
                st.dataframe(pd.DataFrame(hist).tail(20), use_container_width=True, hide_index=True,
                             height=H - 20)
                st.button("Clear", on_click=clear_hist, key="nf_clear")
            else:
                st.caption("Press Save to keep a prediction here and compare different inputs.")


# ==================================================================
# TAB 5: LIVE FLOW (single flow analyzer + what-if)
# ==================================================================
def rand_flow():
    cls = st.session_state.get("lf_cls", "Any")
    n = len(sample) if cls == "Any" else int((sample["Class"] == cls).sum())
    st.session_state["lf_idx"] = int(np.random.randint(0, n))


def reset_wi():
    for f in imp_series.head(5).index:
        st.session_state[f"wi_{f}"] = 1.0


def tab_live():
    top5 = list(imp_series.head(5).index)
    c1, c2, c3 = st.columns([1, 1, 1.3])

    with c1:
        with st.container(border=True):
            ptitle("SELECT A FLOW")
            cls = st.radio("Traffic type", ["Any", "ATTACK", "BENIGN"], horizontal=True, key="lf_cls")
            pool = sample if cls == "Any" else sample[sample["Class"] == cls]
            n = len(pool)
            if st.session_state.get("lf_idx", 0) > n - 1:
                st.session_state["lf_idx"] = 0
            idx = st.slider("Flow number", 0, n - 1, 0, key="lf_idx")
            st.button("🎲 Random flow", on_click=rand_flow, key="lf_rand")
            row = pool.iloc[idx]
            x = row[features].astype(float).to_frame().T
            p = float(model.predict_proba(x)[0, 1])
            is_att = p >= 0.5
            verdict = "ATTACK" if is_att else "BENIGN"
            vcol = RED if is_att else GRN
            ok = is_att == (row["Class"] == "ATTACK")
            st.markdown(f'<div class="verdict" style="color:{vcol}">{verdict}</div>', unsafe_allow_html=True)
            st.caption(f"Actual: {row['Class']}  |  " + ("Model is correct" if ok else "Model is wrong here"))

    with c2:
        with st.container(border=True):
            ptitle("ATTACK PROBABILITY")
            fig = go.Figure(go.Indicator(
                mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 34}},
                gauge={"axis": {"range": [0, 100]},
                       "steps": [{"range": [0, 50], "color": "rgba(52,211,153,0.35)"},
                                 {"range": [50, 100], "color": "rgba(255,107,129,0.35)"}],
                       "bar": {"color": CY}}))
            show(fig, H + 60, False, "lf_g")

    with c3:
        with st.container(border=True):
            ptitle("HIGHER / LOWER THAN NORMAL TRAFFIC")
            z = ((x.iloc[0] - b_mean) / b_std).clip(-10, 10)
            top = z[imp_series.head(10).index].sort_values()
            fig = go.Figure(go.Bar(x=top.values, y=top.index, orientation="h",
                                   marker_color=[RED if v > 0 else BL for v in top.values]))
            fig.update_yaxes(tickfont_size=9, automargin=True, dtick=1)
            fig.update_xaxes(title_text="red = higher than normal, blue = lower", title_font_size=9)
            show(fig, H + 60, False, "lf_d")

    w1, w2 = st.columns([1.2, 1])

    with w1:
        with st.container(border=True):
            ptitle("WHAT-IF: scale the top 5 features (1.0 = original)")
            x2 = x.copy()
            sc1, sc2 = st.columns(2)
            for i, f in enumerate(top5):
                holder = sc1 if i < 3 else sc2
                mult = holder.slider(f[:28], 0.0, 5.0, 1.0, 0.1, key=f"wi_{f}")
                x2[f] = x[f].values * mult
            p2 = float(model.predict_proba(x2)[0, 1])
            m1, m2, m3 = st.columns([1, 1, 1])
            m1.metric("Original", f"{p:.1%}")
            m2.metric("After changes", f"{p2:.1%}", f"{(p2 - p) * 100:+.1f} pts", delta_color="inverse")
            m3.button("Reset sliders", on_click=reset_wi, key="wi_reset")

    with w2:
        with st.container(border=True):
            ptitle("SENSITIVITY: probability as one feature changes")
            f = st.selectbox("Feature", top5, key="se_feat", label_visibility="collapsed")
            mults = np.linspace(0, 5, 26)
            xs = pd.concat([x] * len(mults), ignore_index=True)
            xs[f] = x[f].values[0] * mults
            probs = model.predict_proba(xs)[:, 1] * 100
            fig = go.Figure(go.Scatter(x=mults, y=probs, mode="lines+markers",
                                       line=dict(color=CY, width=2.5), marker_size=5))
            fig.add_vline(x=st.session_state.get(f"wi_{f}", 1.0), line_dash="dash", line_color=AMB)
            fig.update_yaxes(range=[0, 100], title_text="Attack probability %", title_font_size=9)
            fig.update_xaxes(title_text="Multiplier on the feature", title_font_size=9)
            show(fig, H - 10, False, "lf_s")


# ==================================================================
# TAB 6: BATCH SCAN (sample rows or uploaded CSV)
# ==================================================================
def bump_seed():
    st.session_state["seed"] = st.session_state.get("seed", 0) + 1


def tab_batch():
    with st.container(border=True):
        a, b, c, d = st.columns([1.2, 1.4, 1.6, 1])
        source = a.radio("Data source", ["Sample rows", "Upload CSV"], horizontal=True, key="bt_src")
        thr = b.slider("Attack threshold", 0.05, 0.95, 0.5, 0.05, key="bt_thr")
        df = None
        if source == "Sample rows":
            n = c.slider("Number of rows", 100, 5000, 1000, 100, key="bt_n")
            d.button("🔄 Resample", on_click=bump_seed, key="bt_rs")
            df = sample.sample(n, random_state=st.session_state.get("seed", 0))
        else:
            up = c.file_uploader("CSV with the same features", type="csv", key="bt_up",
                                 label_visibility="collapsed")
            if up is not None:
                df = pd.read_csv(up)
                df.columns = df.columns.str.strip()
        dl_slot = d.container()

    if df is None:
        st.info("Upload a CSV to start.")
        return
    missing = [col for col in features if col not in df.columns]
    if missing:
        st.error(f"Missing columns: {missing[:5]} ...")
        return

    X = df[features].replace([np.inf, -np.inf], np.nan).dropna()
    proba = model.predict_proba(X)[:, 1]
    pred = (proba >= thr).astype(int)
    out = df.loc[X.index].copy()
    out["Attack probability"] = proba.round(4)
    out["Prediction"] = np.where(pred == 1, "ATTACK", "BENIGN")

    with dl_slot:
        st.download_button("⬇️ Results CSV", out.to_csv(index=False), "predictions.csv", "text/csv", key="bt_dl")

    items = [("Flows scanned", f"{len(out):,}", CY), ("Attacks detected", f"{int(pred.sum()):,}", RED),
             ("Attack rate", f"{pred.mean():.1%}", AMB)]
    if "Label" in out.columns and pd.api.types.is_numeric_dtype(out["Label"]):
        items.append(("Accuracy on this batch", f"{(pred == out['Label'].values).mean():.2%}", GRN))
    chips(items)

    p1, p2, p3 = st.columns(3)
    with p1:
        with st.container(border=True):
            ptitle("PROBABILITY DISTRIBUTION")
            fig = px.histogram(out, x="Attack probability", color="Prediction", nbins=40, barmode="overlay",
                               color_discrete_map={"ATTACK": RED, "BENIGN": BL}, opacity=0.8)
            fig.add_vline(x=thr, line_dash="dash", line_color=AMB)
            show(fig, H, "top", "bt1")
    with p2:
        with st.container(border=True):
            ptitle("PREDICTED SPLIT")
            cnt = out["Prediction"].value_counts()
            fig = go.Figure(go.Pie(labels=cnt.index, values=cnt.values, hole=0.58, textinfo="label+percent",
                                   marker=dict(colors=[COLORS[k] for k in cnt.index])))
            show(fig, H, False, "bt2")
    with p3:
        with st.container(border=True):
            ptitle("ATTACK RATE TREND (rising or falling)")
            w = max(10, len(pred) // 20)
            roll = pd.Series(pred).rolling(w, min_periods=1).mean() * 100
            fig = go.Figure(go.Scatter(x=np.arange(len(roll)), y=roll.values, mode="lines",
                                       line=dict(color=CY, width=2.5), fill="tozeroy",
                                       fillcolor="rgba(56,189,248,0.15)"))
            fig.update_yaxes(title_text="Attack rate %", title_font_size=9)
            fig.update_xaxes(title_text="Flow order", title_font_size=9)
            show(fig, H, False, "bt3")

    q1, q2 = st.columns([1.2, 1])
    with q1:
        with st.container(border=True):
            ptitle("FEATURES THAT RISE / FALL IN DETECTED ATTACKS")
            if 0 < pred.sum() < len(pred):
                zb = ((X - b_mean) / b_std).clip(-10, 10)
                diff = (zb[pred == 1].mean() - zb[pred == 0].mean())[imp_series.head(25).index]
                diff = diff.reindex(diff.abs().sort_values(ascending=False).head(10).index).sort_values()
                fig = go.Figure(go.Bar(x=diff.values, y=diff.index, orientation="h",
                                       marker_color=[RED if v > 0 else BL for v in diff.values]))
                fig.update_yaxes(tickfont_size=9, automargin=True, dtick=1)
                show(fig, H, False, "bt4")
            else:
                st.info("Needs both ATTACK and BENIGN predictions to compare. Change the threshold.")
    with q2:
        with st.container(border=True):
            ptitle("MOST SUSPICIOUS FLOWS")
            cols = ["Prediction", "Attack probability"] + list(imp_series.head(4).index)
            st.dataframe(out.sort_values("Attack probability", ascending=False)[cols].head(100),
                         use_container_width=True, height=H, hide_index=True)


# ------------------------------------------------------------------
# Render tabs
# ------------------------------------------------------------------
with tabs[0]:
    tab_overview()
with tabs[1]:
    tab_features()
with tabs[2]:
    tab_threshold()
with tabs[3]:
    tab_new()
with tabs[4]:
    tab_live()
with tabs[5]:
    tab_batch()