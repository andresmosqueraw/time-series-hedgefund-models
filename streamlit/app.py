import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

st.set_page_config(
    page_title="Hedge Fund Forecasting",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Datos del paper ────────────────────────────────────────────────────────────
EXPERIMENTS = [
    {"id": "MLP",    "score": 0.0634, "label": "MLP",    "desc": "Red neuronal densa con embeddings (25% entidades)"},
    {"id": "LSTM",   "score": 0.0891, "label": "LSTM",   "desc": "Red recurrente con embeddings (25% entidades)"},
    {"id": "Exp 1",  "score": 0.2675, "label": "Exp 1",  "desc": "LightGBM multi-horizonte + 183 features causales"},
    {"id": "Exp 3",  "score": 0.2712, "label": "Exp 3",  "desc": "LightGBM + Polars + filtro Hampel + interacciones"},
    {"id": "Exp 4",  "score": 0.2824, "label": "Exp 4",  "desc": "Stacking LGB + XGB + Ridge, pesos por recencia"},
    {"id": "Exp 5",  "score": 0.3212, "label": "Exp 5",  "desc": "LightGBM minimalista + IDs jerárquicos como features"},
    {"id": "Exp 6",  "score": 0.3377, "label": "Exp 6",  "desc": "LightGBM minimalista + umbral validación ajustado"},
    {"id": "Exp 7",  "score": 0.3429, "label": "Exp 7 ⭐", "desc": "Ensemble dual: horizonte + sub-categoría, mezcla p^2.5"},
]

HORIZON_SCORES = {1: 0.0731, 3: 0.1289, 10: 0.2510, 25: 0.3271}
SUBCAT_SCORES  = {
    "DPPUO5X2": 0.2126,
    "NQ58FVQM": 0.2367,
    "PHHHVYZI": 0.2879,
    "PZ9S1Z4V": 0.3133,
    "V8BKY1IV": 0.2203,
}

TARGET_STATS = {
    1:  {"mean": -0.083,  "std": 11.700,  "skew": 5.28,  "kurt": 528.5},
    3:  {"mean": -0.252,  "std": 19.361,  "skew": 2.01,  "kurt": 243.8},
    10: {"mean": -0.776,  "std": 33.842,  "skew": 0.83,  "kurt": 176.8},
    25: {"mean": -1.682,  "std": 52.823,  "skew": 0.85,  "kurt": 141.5},
}

EXP_DETAILS = {
    "MLP": {
        "tipo": "Red Neuronal",
        "capas": "[2048, 1024, 512, 256, 128, 64]",
        "optimizador": "Adam (lr=1e-3)",
        "epochs": 50,
        "batch": 4096,
        "val_score": "0.0634",
        "por_qué_falla": "Submuestreo 25%, curtosis extrema (>500 en h=1), sin features temporales.",
    },
    "LSTM": {
        "tipo": "Red Recurrente",
        "capas": "LSTM(256) → Dense(128) → Dense(64) → Dense(1)",
        "optimizador": "Adam (lr=1e-5)",
        "epochs": 100,
        "batch": 256,
        "val_score": "0.0891",
        "por_qué_falla": "Mismo submuestreo 25%, sin información de orden temporal explícita.",
    },
    "Exp 1": {
        "tipo": "LightGBM",
        "features": 183,
        "semillas": 5,
        "modelos": "4 (uno por horizonte)",
        "val_score_h1": 0.075,
        "val_score_h25": 0.293,
        "public_score": "0.2675",
    },
    "Exp 3": {
        "tipo": "LightGBM + Polars",
        "novedad": "Filtro Hampel, selección de features en 2 pasadas",
        "features_h1": 197, "features_h25": 217,
        "public_score": "0.2712",
    },
    "Exp 4": {
        "tipo": "Stacking LGB + XGB + Ridge",
        "novedad": "Pesos por recencia (40×), meta-modelo Ridge con pesos no negativos",
        "peso_lgb_h10": "94.3%",
        "public_score": "0.2824",
    },
    "Exp 5": {
        "tipo": "LightGBM Minimalista",
        "novedad": "code, sub_code, sub_category, horizon como dtype='category'",
        "iters_h1": 133, "iters_h25": 117,
        "public_score": "0.3212",
    },
    "Exp 6": {
        "tipo": "LightGBM Minimalista (determinista)",
        "novedad": "Umbral validación 3500 → 3503, seed=5, deterministic=True",
        "iters_h10": 541, "iters_h25": 433,
        "public_score": "0.3377",
    },
    "Exp 7": {
        "tipo": "Ensemble Dual LightGBM",
        "modelos_horizonte": 4,
        "modelos_subcategoria": 5,
        "blend_power": 2.5,
        "peso_pool_b": "67.1%",
        "public_score": "0.3429",
    },
}

# ── Helpers ───────────────────────────────────────────────────────────────────
def weighted_rmse_score(y_true, y_pred, weights):
    y_true   = np.asarray(y_true, dtype=float)
    y_pred   = np.asarray(y_pred, dtype=float)
    weights  = np.asarray(weights, dtype=float)
    num      = np.sum(weights * (y_true - y_pred) ** 2)
    den      = np.sum(weights * y_true ** 2)
    if den <= 0:
        return 0.0
    ratio = np.clip(num / den, 0.0, 1.0)
    return float(np.sqrt(1.0 - ratio))


def blend(h_score, s_score, h_pred, s_pred, power):
    ph = h_score ** power
    ps = s_score ** power
    total = ph + ps
    if total == 0:
        return 0.5, 0.5, (h_pred + s_pred) / 2
    wh = ph / total
    ws = ps / total
    return wh, ws, wh * h_pred + ws * s_pred


# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.title("📈 Hedge Fund Forecasting")
st.sidebar.markdown("**Equipo Arthur XXIV** · Uniandes 2025")
st.sidebar.divider()
pagina = st.sidebar.radio(
    "Navegar a:",
    ["El Problema", "Los Datos", "Experimentos", "Demo: Ensemble Dual", "Métrica Interactiva"],
    index=0,
)
st.sidebar.divider()
st.sidebar.caption("Competencia Kaggle · Score final: **0.3429**")

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1: EL PROBLEMA
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "El Problema":
    st.title("¿Puede una máquina adivinar el futuro del dinero?")
    st.markdown("### Hedge Fund Time Series Forecasting — Kaggle Competition")
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric("Filas entrenamiento", "5.3 M")
    col2.metric("Filas a predecir",    "1.4 M")
    col3.metric("Features anónimas",   "86")

    col4, col5, col6 = st.columns(3)
    col4.metric("Horizontes",          "1 · 3 · 10 · 25")
    col5.metric("Score objetivo",      "> 0.30")
    col6.metric("Score alcanzado",     "0.3429", delta="+0.0429")

    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("El reto")
        st.markdown("""
Un **fondo de cobertura** (Hedge Fund) activo aporta datos financieros completamente anonimizados.
La tarea: **predecir `y_target`** para cada fila del conjunto de test.

- Los datos tienen una señal extremadamente débil (*SNR ≈ 0.063*)
- No se conoce el significado de ninguna columna ni las fechas exactas
- La única certeza es el orden temporal: `ts_index` va de 1 a 4 376

> *Como intentar escuchar un susurro en un estadio lleno de gente.*
        """)

        st.error("⛔ **Regla de oro:** solo puedes usar datos del pasado (`ts_index ≤ t`). Usar el futuro es trampa y la competencia lo penaliza severamente.")

    with col_b:
        st.subheader("La métrica")
        st.latex(r"\text{Score} = \sqrt{1 - \text{clip}\!\left(\frac{\sum w_i(y_i - \hat{y}_i)^2}{\sum w_i y_i^2},\; 0,\; 1\right)}")
        st.markdown("""
| Valor | Significado |
|-------|-------------|
| **0** | Tan malo como predecir siempre 0 |
| **0.30** | Meta de la competencia |
| **1** | Predicción perfecta |

La métrica pondera más los errores en muestras de mayor importancia (`weight`).
        """)

    st.divider()
    st.subheader("Línea de tiempo")
    timeline_data = {
        "ts_index": ["1", "3 500", "3 503", "4 376"],
        "Rol":      ["Inicio entrenamiento", "Corte train/val (Exp 5)", "Corte train/val (Exp 6)", "Fin test"],
    }
    fig_tl = go.Figure()
    xs = [1, 3500, 3503, 4376]
    labels = ["Inicio\ntrain", "Corte val\n(Exp 5)", "Corte val\n(Exp 6)", "Fin\ntest"]
    colors = ["green", "orange", "red", "gray"]
    fig_tl.add_trace(go.Scatter(
        x=xs, y=[0]*4,
        mode="markers+text",
        marker=dict(size=18, color=colors),
        text=labels, textposition="top center",
    ))
    fig_tl.add_shape(type="rect", x0=1, x1=3500, y0=-0.3, y1=0.3,
                     fillcolor="rgba(0,200,0,0.1)", line_width=0)
    fig_tl.add_shape(type="rect", x0=3500, x1=4376, y0=-0.3, y1=0.3,
                     fillcolor="rgba(255,165,0,0.1)", line_width=0)
    fig_tl.add_annotation(x=1750, y=-0.25, text="TRAIN", showarrow=False, font_size=14, font_color="green")
    fig_tl.add_annotation(x=3900, y=-0.25, text="TEST", showarrow=False, font_size=14, font_color="orange")
    fig_tl.update_layout(
        height=200, showlegend=False,
        xaxis=dict(title="ts_index"), yaxis=dict(visible=False),
        margin=dict(t=30, b=30),
    )
    st.plotly_chart(fig_tl, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2: LOS DATOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Los Datos":
    st.title("Los Datos")
    st.markdown("Dataset aportado por un fondo de cobertura activo (Kaggle). **Todo anonimizado.**")
    st.divider()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Diccionario de columnas")
        dict_df = pd.DataFrame([
            {"Columna": "id",           "Tipo": "str",   "Descripción": "Clave única: code__sub_code__sub_category__horizon__ts_index"},
            {"Columna": "code",         "Tipo": "str",   "Descripción": "Entidad primaria (anonimizada)"},
            {"Columna": "sub_code",     "Tipo": "str",   "Descripción": "Subgrupo dentro de code"},
            {"Columna": "sub_category", "Tipo": "str",   "Descripción": "Categoría amplia (5 valores)"},
            {"Columna": "ts_index",     "Tipo": "int",   "Descripción": "Eje temporal entero — orden causal estricto"},
            {"Columna": "horizon",      "Tipo": "int",   "Descripción": "Régimen: 1 (corto), 3, 10, 25 (extra-largo)"},
            {"Columna": "weight",       "Tipo": "float", "Descripción": "Peso por muestra — solo para la métrica, nunca feature"},
            {"Columna": "y_target",     "Tipo": "float", "Descripción": "Variable objetivo (solo en train)"},
            {"Columna": "feature_a…ch", "Tipo": "float", "Descripción": "86 features numéricas anónimas, admiten nulos"},
        ])
        st.dataframe(dict_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Estadísticas de y_target por horizonte")
        stat_rows = []
        for h, s in TARGET_STATS.items():
            stat_rows.append({"Horizonte": f"h={h}", **s})
        stat_df = pd.DataFrame(stat_rows).rename(columns={
            "mean": "Media", "std": "Desv. Est.", "skew": "Asimetría", "kurt": "Curtosis"
        })
        st.dataframe(stat_df, use_container_width=True, hide_index=True)

        st.caption("La curtosis altísima en h=1 (528) indica valores extremos que dominan el error.")

    st.divider()
    st.subheader("Dificultad por horizonte — Curtosis vs Score obtenido")

    fig_diff = go.Figure()
    horizons = list(TARGET_STATS.keys())
    kurts    = [TARGET_STATS[h]["kurt"] for h in horizons]
    scores   = [HORIZON_SCORES[h] for h in horizons]

    fig_diff.add_trace(go.Bar(
        name="Curtosis (eje izq.)",
        x=[f"h={h}" for h in horizons],
        y=kurts,
        marker_color=["#EF553B","#FF7F0E","#FFA500","#2CA02C"],
        yaxis="y",
    ))
    fig_diff.add_trace(go.Scatter(
        name="Score validación (eje der.)",
        x=[f"h={h}" for h in horizons],
        y=scores,
        mode="lines+markers+text",
        text=[f"{s:.3f}" for s in scores],
        textposition="top center",
        marker=dict(size=12, color="royalblue"),
        line=dict(color="royalblue", width=3),
        yaxis="y2",
    ))
    fig_diff.update_layout(
        height=380,
        yaxis=dict(title="Curtosis", side="left"),
        yaxis2=dict(title="Score validación", side="right", overlaying="y", range=[0, 0.5]),
        legend=dict(orientation="h", y=1.12),
        margin=dict(t=50, b=30),
    )
    st.plotly_chart(fig_diff, use_container_width=True)
    st.info("A mayor curtosis (más valores extremos) → menor score. El horizonte h=1 es el más difícil.")

    st.divider()
    st.subheader("¿Por qué es tan difícil? SNR extremadamente bajo")
    snr_val = 0.063
    col3, col4 = st.columns([1, 2])
    with col3:
        st.metric("Signal-to-Noise Ratio (baseline)", f"{snr_val:.3f}")
        st.markdown("Para comparar: señal de voz en llamada telefónica ≈ 0.8–1.0")
    with col4:
        fig_snr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=snr_val,
            title={"text": "SNR"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": "firebrick"},
                "steps": [
                    {"range": [0, 0.1], "color": "#ffcccc"},
                    {"range": [0.1, 0.5], "color": "#ffffcc"},
                    {"range": [0.5, 1.0], "color": "#ccffcc"},
                ],
                "threshold": {"line": {"color": "red", "width": 4}, "value": 0.063},
            },
        ))
        fig_snr.update_layout(height=250, margin=dict(t=20, b=20))
        st.plotly_chart(fig_snr, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3: EXPERIMENTOS
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Experimentos":
    st.title("Los 8 Experimentos — Evolución del Score")
    st.divider()

    labels = [e["label"] for e in EXPERIMENTS]
    scores = [e["score"] for e in EXPERIMENTS]
    colors = ["#EF553B" if s < 0.15 else "#FF7F0E" if s < 0.30 else "#2CA02C" for s in scores]
    colors[-1] = "#1f77b4"

    fig_exp = go.Figure()
    fig_exp.add_shape(type="line", x0=-0.5, x1=7.5, y0=0.30, y1=0.30,
                      line=dict(color="gold", width=2, dash="dash"))
    fig_exp.add_annotation(x=7.3, y=0.305, text="Meta: 0.30", showarrow=False,
                           font=dict(color="gold", size=13))
    fig_exp.add_trace(go.Bar(
        x=labels, y=scores,
        marker_color=colors,
        text=[f"{s:.4f}" for s in scores],
        textposition="outside",
    ))
    fig_exp.update_layout(
        height=420,
        yaxis=dict(title="Score público (Kaggle)", range=[0, 0.42]),
        xaxis_title="Experimento",
        margin=dict(t=40, b=30),
        showlegend=False,
    )
    st.plotly_chart(fig_exp, use_container_width=True)

    st.divider()
    st.subheader("Detalle de cada experimento")

    tab_labels = [e["label"] for e in EXPERIMENTS]
    tabs = st.tabs(tab_labels)

    for i, (tab, exp) in enumerate(zip(tabs, EXPERIMENTS)):
        with tab:
            d = EXP_DETAILS.get(exp["id"], {})
            col_l, col_r = st.columns([1, 1])
            with col_l:
                st.markdown(f"**Descripción:** {exp['desc']}")
                st.metric("Score público", exp["score"])
                if "tipo" in d:
                    st.markdown(f"**Tipo:** {d['tipo']}")
                if "novedad" in d:
                    st.info(f"**Novedad clave:** {d['novedad']}")
                if "por_qué_falla" in d:
                    st.error(f"**¿Por qué falló?** {d['por_qué_falla']}")
            with col_r:
                if exp["id"] == "MLP":
                    st.markdown("**Arquitectura:**")
                    arch = pd.DataFrame({
                        "Capa": ["Embeddings", "Linear 2048", "Linear 1024", "Linear 512", "Linear 256", "Linear 128", "Linear 64", "Output"],
                        "Activación": ["—", "ReLU+BN+Drop", "ReLU+BN+Drop", "ReLU+BN+Drop", "ReLU+BN+Drop", "ReLU+BN+Drop", "ReLU+BN+Drop", "—"],
                    })
                    st.dataframe(arch, use_container_width=True, hide_index=True)
                elif exp["id"] == "LSTM":
                    st.markdown("**Arquitectura:**")
                    arch = pd.DataFrame({
                        "Capa": ["Embeddings x4", "LSTM(256)", "Dense(128)", "Dropout(0.2)", "Dense(64)", "Dense(1)"],
                        "Detalle": ["dim=86 cada una", "1 capa", "relu", "—", "relu", "—"],
                    })
                    st.dataframe(arch, use_container_width=True, hide_index=True)
                elif exp["id"] == "Exp 1":
                    st.markdown("**Scores por horizonte (validación local):**")
                    h_df = pd.DataFrame([
                        {"Horizonte": f"h={h}", "Score": round(s, 5)}
                        for h, s in {1: 0.07547, 3: 0.13232, 10: 0.23661, 25: 0.29283}.items()
                    ])
                    st.dataframe(h_df, use_container_width=True, hide_index=True)
                elif exp["id"] == "Exp 4":
                    st.markdown("**Pesos Ridge por horizonte:**")
                    w_df = pd.DataFrame([
                        {"h": 1,  "w_LGB": "78.6%", "w_XGB": "21.4%", "Score stack": 0.081},
                        {"h": 3,  "w_LGB": "64.4%", "w_XGB": "35.6%", "Score stack": 0.136},
                        {"h": 10, "w_LGB": "94.3%", "w_XGB": "5.7%",  "Score stack": 0.244},
                        {"h": 25, "w_LGB": "90.6%", "w_XGB": "9.4%",  "Score stack": 0.278},
                    ])
                    st.dataframe(w_df, use_container_width=True, hide_index=True)
                elif exp["id"] == "Exp 7":
                    st.markdown("**Scores de validación por modelo:**")
                    rows = []
                    for h, s in HORIZON_SCORES.items():
                        rows.append({"Pool": "Horizonte", "Modelo": f"h={h}", "Score": round(s, 4)})
                    for sc, s in SUBCAT_SCORES.items():
                        rows.append({"Pool": "Sub-categoría", "Modelo": sc, "Score": round(s, 4)})
                    e7_df = pd.DataFrame(rows)
                    st.dataframe(e7_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Las 3 lecciones clave")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.success("**LightGBM gana 4× a redes neuronales** en datos tabulares financieros de bajo SNR.")
    with c2:
        st.success("**La identidad importa más que las features.** Pasar IDs como `category` fue el mayor salto (+0.04).")
    with c3:
        st.success("**Más complejidad ≠ mejor.** El modelo minimalista superó al stacking LGB+XGB+Ridge.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4: DEMO ENSEMBLE DUAL
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Demo: Ensemble Dual":
    st.title("Demo: Ensemble Dual — Cómo se mezclan las predicciones")
    st.markdown("""
El experimento ganador entrena **9 modelos** y mezcla sus predicciones:
- **Pool A:** 4 modelos (uno por horizonte h=1, 3, 10, 25)
- **Pool B:** 5 modelos (uno por sub-categoría)
    """)
    st.divider()

    col_ctrl, col_res = st.columns([1, 1])

    with col_ctrl:
        st.subheader("Parámetros de la fila a predecir")
        horizonte   = st.selectbox("Horizonte", [1, 3, 10, 25], index=2)
        subcategoria = st.selectbox("Sub-categoría", list(SUBCAT_SCORES.keys()), index=3)
        h_pred      = st.slider("Predicción Pool A (horizonte)", -20.0, 20.0, 0.5, 0.1)
        s_pred      = st.slider("Predicción Pool B (sub-cat)",   -20.0, 20.0, 2.5, 0.1)
        power       = st.slider("Potencia de mezcla (blend_power)", 0.5, 5.0, 2.5, 0.1)

    h_score = HORIZON_SCORES[horizonte]
    s_score = SUBCAT_SCORES[subcategoria]
    wh, ws, final_pred = blend(h_score, s_score, h_pred, s_pred, power)

    with col_res:
        st.subheader("Cálculo paso a paso")
        st.markdown(f"""
**Scores de validación:**
- Pool A (h={horizonte}): `{h_score:.4f}`
- Pool B ({subcategoria}): `{s_score:.4f}`

**Potencia aplicada (p = {power}):**
- Pool A: `{h_score:.4f}^{power}` = `{h_score**power:.6f}`
- Pool B: `{s_score:.4f}^{power}` = `{s_score**power:.6f}`
- Total: `{h_score**power + s_score**power:.6f}`

**Pesos resultantes:**
- Pool A: `{wh*100:.1f}%`
- Pool B: `{ws*100:.1f}%`
        """)
        st.divider()
        st.metric("Predicción Pool A", f"{h_pred:.3f}")
        st.metric("Predicción Pool B", f"{s_pred:.3f}")
        st.metric("Predicción FINAL", f"{final_pred:.4f}",
                  delta=f"Pool B domina {ws*100:.0f}%" if ws > wh else f"Pool A domina {wh*100:.0f}%")

    st.divider()

    # Gráfico de pesos vs potencia
    st.subheader(f"¿Cómo cambia el peso de cada pool con la potencia? (h={horizonte}, sub_cat={subcategoria})")
    powers = np.linspace(0.1, 6.0, 200)
    whs, wss = [], []
    for p in powers:
        wh_p, ws_p, _ = blend(h_score, s_score, 0, 0, p)
        whs.append(wh_p * 100)
        wss.append(ws_p * 100)

    fig_blend = go.Figure()
    fig_blend.add_trace(go.Scatter(x=powers, y=whs, name=f"Pool A (h={horizonte})",
                                   line=dict(color="royalblue", width=2)))
    fig_blend.add_trace(go.Scatter(x=powers, y=wss, name=f"Pool B ({subcategoria})",
                                   line=dict(color="darkorange", width=2)))
    fig_blend.add_vline(x=power, line_dash="dash", line_color="gray",
                        annotation_text=f"p={power}", annotation_position="top right")
    fig_blend.update_layout(
        xaxis_title="Potencia (blend_power)",
        yaxis_title="Peso (%)",
        height=340,
        legend=dict(orientation="h", y=1.1),
        margin=dict(t=40, b=30),
        yaxis=dict(range=[0, 100]),
    )
    st.plotly_chart(fig_blend, use_container_width=True)

    st.divider()

    # Mapa de calor de pesos
    st.subheader("Pesos de Pool B (sub-cat) según horizonte y sub-categoría (power=2.5)")
    heatmap_data = []
    for h, hs in HORIZON_SCORES.items():
        row = []
        for sc, ss in SUBCAT_SCORES.items():
            _, ws_val, _ = blend(hs, ss, 0, 0, 2.5)
            row.append(round(ws_val * 100, 1))
        heatmap_data.append(row)

    fig_hm = go.Figure(go.Heatmap(
        z=heatmap_data,
        x=list(SUBCAT_SCORES.keys()),
        y=[f"h={h}" for h in HORIZON_SCORES.keys()],
        colorscale="RdYlGn",
        text=[[f"{v}%" for v in row] for row in heatmap_data],
        texttemplate="%{text}",
        colorbar=dict(title="Peso Pool B (%)"),
    ))
    fig_hm.update_layout(
        height=300,
        xaxis_title="Sub-categoría",
        yaxis_title="Horizonte",
        margin=dict(t=20, b=30),
    )
    st.plotly_chart(fig_hm, use_container_width=True)
    st.caption("Verde = Pool B domina. Rojo = Pool A domina. En h=1 con score 0.073, Pool B siempre domina.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5: MÉTRICA INTERACTIVA
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "Métrica Interactiva":
    st.title("Métrica Interactiva — Weighted RMSE Skill Score")
    st.latex(r"\text{Score} = \sqrt{1 - \text{clip}\!\left(\frac{\sum w_i(y_i - \hat{y}_i)^2}{\sum w_i y_i^2},\; 0,\; 1\right)}")
    st.divider()

    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Simulador de predicciones")
        n = st.slider("Número de observaciones", 3, 10, 5)
        st.markdown("**Ajusta cada valor:**")
        y_true_vals, y_pred_vals, w_vals = [], [], []
        for i in range(n):
            c1, c2, c3 = st.columns(3)
            yt = c1.number_input(f"y_true[{i}]",  value=round(np.random.uniform(-10, 10), 1), key=f"yt{i}", step=0.5)
            yp = c2.number_input(f"y_pred[{i}]",  value=round(yt + np.random.uniform(-3, 3), 1), key=f"yp{i}", step=0.5)
            w  = c3.number_input(f"weight[{i}]",  value=1.0, min_value=0.1, max_value=5.0, key=f"w{i}", step=0.1)
            y_true_vals.append(yt)
            y_pred_vals.append(yp)
            w_vals.append(w)

    with col_b:
        st.subheader("Resultado")
        y_true_arr = np.array(y_true_vals)
        y_pred_arr = np.array(y_pred_vals)
        w_arr      = np.array(w_vals)

        score = weighted_rmse_score(y_true_arr, y_pred_arr, w_arr)
        num   = np.sum(w_arr * (y_true_arr - y_pred_arr) ** 2)
        den   = np.sum(w_arr * y_true_arr ** 2)
        ratio = np.clip(num / den, 0, 1) if den > 0 else 0

        st.metric("Score", f"{score:.4f}")
        st.markdown(f"""
**Desglose:**
- Numerador (error²): `{num:.4f}`
- Denominador (varianza²): `{den:.4f}`
- Ratio: `{ratio:.4f}`
- `√(1 − {ratio:.4f})` = `{score:.4f}`
        """)

        color = "green" if score > 0.30 else "orange" if score > 0.10 else "red"
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=score,
            delta={"reference": 0.30, "increasing": {"color": "green"}},
            title={"text": "Score"},
            gauge={
                "axis": {"range": [0, 1]},
                "bar": {"color": color},
                "steps": [
                    {"range": [0, 0.10], "color": "#ffcccc"},
                    {"range": [0.10, 0.30], "color": "#ffffcc"},
                    {"range": [0.30, 1.0], "color": "#ccffcc"},
                ],
                "threshold": {"line": {"color": "gold", "width": 4}, "value": 0.30},
            },
        ))
        fig_gauge.update_layout(height=280, margin=dict(t=20, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

    st.divider()
    st.subheader("Comparativa: nuestros experimentos en contexto")
    exp_df = pd.DataFrame(EXPERIMENTS)[["label", "score", "desc"]].rename(columns={
        "label": "Experimento", "score": "Score público", "desc": "Descripción"
    })
    exp_df["vs Meta 0.30"] = exp_df["Score público"].apply(
        lambda s: "✅ Superado" if s >= 0.30 else "❌ Bajo meta"
    )
    st.dataframe(exp_df, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("¿Qué significa predecir 0 siempre?")
    st.markdown("""
Si predices `ŷ = 0` para todo:

$$\\text{Score} = \\sqrt{1 - \\frac{\\sum w_i \\cdot y_i^2}{\\sum w_i \\cdot y_i^2}} = \\sqrt{1-1} = 0$$

Los experimentos MLP (0.063) y LSTM (0.089) apenas superaron este baseline.
LightGBM logró **0.3429**, lo que significa que captura ~34% de la varianza ponderada.
    """)
