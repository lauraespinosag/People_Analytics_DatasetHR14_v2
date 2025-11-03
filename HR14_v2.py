# app_streamlit_hr_genero_hire.py
# Streamlit: Compañía / Departamento / Individual,
# con comparativa por género y por fecha de contratación (DateofHire)
# Ejecuta: streamlit run app_streamlit_hr_genero_hire.py

from typing import Optional, Dict, Any
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.io as pio
from sklearn.tree import DecisionTreeRegressor
from datetime import datetime

st.set_page_config(page_title="HR — Niveles, Género y Fecha de contratación", layout="wide")

# =========================
# CONFIGURACIÓN
# =========================
DATA_PATH = "HRDataset_v14.csv"
TARGET = "PerfScoreID"   # Este tablero muestra SOLO PREDICCIONES
FEATURES = [
    "EmpSatisfaction",
    "EngagementSurvey",
    "Salary",
    "SpecialProjectsCount",
    "Absences",
    "DaysLateLast30",
]
LEVEL_MIN, LEVEL_MAX = 1, 5
COMPANY_LEVELS_TO_SHOW = [1, 2, 3, 4]

# =========================
# TEMA PLOTLY
# =========================
pio.templates.default = "plotly_white"
DEFAULT_SEQ = px.colors.qualitative.Set2

def style_fig(fig):
    fig.update_layout(paper_bgcolor="white", plot_bgcolor="white")
    return fig

# =========================
# ESTILO (tipografía más pequeña)
# =========================
st.markdown(
    """
    <style>
      html, body, [class*="css"] { font-size: 15px; }
      h1 { font-size: 1.6rem !important; margin-bottom: .2rem; }
      h2 { font-size: 1.3rem !important; margin-bottom: .2rem; }
      h3 { font-size: 1.1rem !important; margin-bottom: .2rem; }
      .stTabs [data-baseweb="tab-list"] { gap: 4px; }
      .stTabs [data-baseweb="tab"] { padding: 8px 12px; border-radius: 10px; }
      .block-container { padding-top: 1.2rem; padding-bottom: 1.2rem; }
      .notes { background: #f8fafc; border: 1px solid #e5e7eb; border-radius:12px; padding:12px; }
      .foot { color:#6b7280; font-size:.85rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# CARGA DE DATOS
# =========================
df = pd.read_csv(DATA_PATH)

missing = [c for c in FEATURES if c not in df.columns]
if missing:
    st.error(f"Faltan columnas requeridas en el CSV: {missing}")
    st.stop()

if "Department" not in df.columns:
    st.error("El CSV no tiene la columna 'Department'.")
    st.stop()

# ---- Género (usa el dataset real) ----
GENDER_CANDIDATES = ["Sex", "Gender", "GenderID"]
gender_col: Optional[str] = None
for c in GENDER_CANDIDATES:
    if c in df.columns:
        gender_col = c
        break

# ---- Fecha de contratación (DateofHire) ----
HIRE_COL = "DateofHire" if "DateofHire" in df.columns else None

if HIRE_COL is not None:
    df["DateofHire_dt"] = pd.to_datetime(df[HIRE_COL], errors="coerce")
    df["HireYear"] = df["DateofHire_dt"].dt.year

    if "LastPerformanceReview_Date" in df.columns:
        ref_date = pd.to_datetime(df["LastPerformanceReview_Date"], errors="coerce").max()
    else:
        ref_date = df["DateofHire_dt"].max()

    df["TenureYears"] = ((ref_date - df["DateofHire_dt"]).dt.days / 365.25).clip(lower=0)
else:
    df["DateofHire_dt"] = pd.NaT
    df["HireYear"] = np.nan
    df["TenureYears"] = np.nan

# =========================
# MODELO (árbol sencillo)
# =========================
X = df[FEATURES].copy()
if TARGET in df.columns:
    y = df[TARGET].astype(float)
else:
    y = np.full(len(df), X[FEATURES[0]].mean())

tree = DecisionTreeRegressor(max_depth=5, min_samples_leaf=2, random_state=42)
tree.fit(X, y)

def to_level(pred: np.ndarray) -> np.ndarray:
    arr = np.rint(pred).astype(int)
    return np.clip(arr, LEVEL_MIN, LEVEL_MAX)

# Baseline global
df["_pred_base"] = tree.predict(df[FEATURES])
df["_level_base"] = to_level(df["_pred_base"])

# Identificador principal
id_cols = [c for c in ["EmpID", "Employee_Name"] if c in df.columns]
id_col = id_cols[0] if id_cols else None
if id_col is None:
    df["_row_id"] = df.index
    id_col = "_row_id"

# =========================
# ESTADO — NOTAS POR DEPARTAMENTO
# =========================
if "dept_notes" not in st.session_state:
    st.session_state.dept_notes: Dict[str, Dict[str, Any]] = {}

def get_dept_notes(dept: str) -> Dict[str, str]:
    return st.session_state.dept_notes.get(dept, {"ideas": "", "sugerencias": "", "last_saved": ""})

def set_dept_notes(dept: str, ideas: str, sugerencias: str) -> None:
    st.session_state.dept_notes[dept] = {
        "ideas": ideas,
        "sugerencias": sugerencias,
        "last_saved": datetime.now().strftime("%Y-%m-%d %H:%M")
    }

# =========================
# PALANCAS GLOBALES (±100% relativo)
# =========================
st.sidebar.header("Palancas globales (±100% relativo)")
st.sidebar.caption("Se aplican como cambios porcentuales sobre los valores actuales del departamento/individuo.")

delta_satisfaction_pct = st.sidebar.slider("EmpSatisfaction (%)",      -100, 100, 0, 1, format="%d%%")
delta_engagement_pct   = st.sidebar.slider("EngagementSurvey (%)",     -100, 100, 0, 1, format="%d%%")
delta_salary_pct       = st.sidebar.slider("Salary (%)",               -100, 100, 0, 1, format="%d%%")
delta_projects_pct     = st.sidebar.slider("SpecialProjectsCount (%)", -100, 100, 0, 1, format="%d%%")
delta_absences_pct     = st.sidebar.slider("Absences (%)",             -100, 100, 0, 1, format="%d%%")
delta_lates_pct        = st.sidebar.slider("DaysLateLast30 (%)",       -100, 100, 0, 1, format="%d%%")

def apply_levers(subset: pd.DataFrame) -> pd.DataFrame:
    """Aplica palancas multiplicativas ±100% al subset (con saneos)."""
    out = subset.copy()

    # Convertir y rellenar NaN con mediana local
    es  = pd.to_numeric(out["EmpSatisfaction"],      errors="coerce").fillna(out["EmpSatisfaction"].median())
    en  = pd.to_numeric(out["EngagementSurvey"],     errors="coerce").fillna(out["EngagementSurvey"].median())
    spc = pd.to_numeric(out["SpecialProjectsCount"], errors="coerce").fillna(out["SpecialProjectsCount"].median())
    absn= pd.to_numeric(out["Absences"],             errors="coerce").fillna(out["Absences"].median())
    late= pd.to_numeric(out["DaysLateLast30"],       errors="coerce").fillna(out["DaysLateLast30"].median())
    sal = pd.to_numeric(out["Salary"],               errors="coerce").fillna(out["Salary"].median())

    # Aplicar % relativos (±100%)
    es   = es   * (1 + delta_satisfaction_pct/100.0)
    en   = en   * (1 + delta_engagement_pct/100.0)
    spc  = spc  * (1 + delta_projects_pct/100.0)
    absn = absn * (1 + delta_absences_pct/100.0)
    late = late * (1 + delta_lates_pct/100.0)
    sal  = sal  * (1 + delta_salary_pct/100.0)

    # Saneos básicos (no negativos)
    es   = es.clip(lower=0)
    en   = en.clip(lower=0)
    spc  = spc.clip(lower=0)
    absn = absn.clip(lower=0)
    late = late.clip(lower=0)
    sal  = sal.clip(lower=0)

    out["EmpSatisfaction"]      = es
    out["EngagementSurvey"]     = en
    out["SpecialProjectsCount"] = spc
    out["Absences"]             = absn
    out["DaysLateLast30"]       = late
    out["Salary"]               = sal
    return out

# =========================
# CONSTRUCCIÓN DEL INFORME HTML
# =========================
def build_report_html(
    dept_name: str,
    dist_tbl: pd.DataFrame,
    fig_dist_local,
    comp_levels_tbl: Optional[pd.DataFrame],
    fig_comp_local,
    ideas_html: str,
    sugerencias_html: str,
    kpis: Dict[str, Any],
) -> str:
    css = """
    <style>
      body { font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; margin: 20px; color:#111827; }
      h1 { font-size: 22px; margin: 0 0 8px 0;}
      h2 { font-size: 18px; margin: 16px 0 6px 0;}
      .kpis { display: grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap: 12px; }
      .card { border:1px solid #e5e7eb; border-radius:12px; padding:12px; background:#fff; }
      .label { color:#6b7280; font-size:12px; }
      .val { font-size: 18px; font-weight:600; }
      .section { margin-top: 16px; }
      .notes { background:#f8fafc; border:1px solid #e5e7eb; border-radius:12px; padding:12px; }
      table { border-collapse: collapse; width: 100%; font-size: 13px; }
      th, td { border: 1px solid #e5e7eb; padding: 6px 8px; text-align: left; }
      th { background:#f3f4f6; }
      .foot { color:#6b7280; font-size:12px; margin-top: 12px; }
    </style>
    """
    title = f"<h1>Informe — Departamento {dept_name}</h1><div class='foot'>Generado: {datetime.now().strftime('%Y-%m-%d %H:%M')}</div>"
    kpis_html = f"""
    <div class="kpis">
      <div class="card"><div class="label">Personas</div><div class="val">{kpis['personas']}</div></div>
      <div class="card"><div class="label">PerfScoreID medio (predicho)</div><div class="val">{kpis['media']:.3f}</div></div>
      <div class="card"><div class="label">Nivel más frecuente</div><div class="val">{kpis['moda']}</div></div>
    </div>
    """

    dist_tbl_html = dist_tbl.to_html(index=False)
    fig_dist_html = pio.to_html(fig_dist_local, include_plotlyjs="cdn", full_html=False, config={"displayModeBar": False})

    comp_section = ""
    if comp_levels_tbl is not None and fig_comp_local is not None:
        comp_tbl_html = comp_levels_tbl.to_html(index=False)
        fig_comp_html = pio.to_html(fig_comp_local, include_plotlyjs=False, full_html=False, config={"displayModeBar": False})
        comp_section = f"""
        <div class="section">
          <h2>Baseline vs Escenario (predicho)</h2>
          {fig_comp_html}
          <div class="section">
            <h2>Tabla comparativa por nivel</h2>
            {comp_tbl_html}
          </div>
        </div>
        """

    ideas_block = ideas_html if ideas_html.strip() else "<i>(sin notas)</i>"
    sugerencias_block = sugerencias_html if sugerencias_html.strip() else "<i>(sin notas)</i>"

    notes_html = f"""
    <div class="section">
      <h2>Ideas de mejora a trabajar</h2>
      <div class="notes">{ideas_block}</div>
    </div>
    <div class="section">
      <h2>Sugerencias de mejora</h2>
      <div class="notes">{sugerencias_block}</div>
    </div>
    """

    return f"<!DOCTYPE html><html><head><meta charset='utf-8'>{css}</head><body>{title}{kpis_html}<div class='section'><h2>Distribución de niveles — Baseline</h2>{fig_dist_html}<div class='section'><h2>Tabla de distribución</h2>{dist_tbl_html}</div></div>{comp_section}{notes_html}<div class='foot'>Este informe utiliza SOLO predicciones del modelo y niveles redondeados a [1,5].</div></body></html>"

# =========================
# CABECERA
# =========================
st.title("📊 HR — Niveles por Compañía, Departamento e Individual")
st.caption("Todo lo mostrado son predicciones del modelo; los niveles se calculan redondeando el PerfScoreID predicho y acotando a [1, 5].")

if gender_col is None:
    st.caption("⚠️ No se ha encontrado ninguna columna de género estándar (por ejemplo, 'Sex', 'Gender').")
else:
    st.caption(f"Columna de género usada: **{gender_col}**")

if HIRE_COL is None:
    st.caption("⚠️ No se ha encontrado la columna 'DateofHire'.")
else:
    st.caption("Se ha añadido comparativa por **fecha de contratación** usando la columna `DateofHire`.")

# =========================
# TABS
# =========================
tab_company, tab_dept, tab_individual = st.tabs(["🏢 Compañía", "🏬 Departamento", "👤 Individual"])

# =========================
# COMPAÑÍA
# =========================
with tab_company:
    st.header("Conteo y exploración por niveles (1–4)")
    colC1, colC2 = st.columns([2, 1])

    mask = df["_level_base"].isin(COMPANY_LEVELS_TO_SHOW)
    dist = df.loc[mask, "_level_base"].value_counts().reindex(COMPANY_LEVELS_TO_SHOW, fill_value=0)
    dist_table = pd.DataFrame({"Nivel": dist.index, "Personas": dist.values})
    dist_table["%"] = (dist_table["Personas"] / max(1, dist_table["Personas"].sum()) * 100).round(1)

    with colC1:
        fig = px.bar(
            dist_table, x="Nivel", y="Personas", text="Personas",
            title="Distribución de personas por nivel (predicho, 1–4)",
            labels={"Nivel": "Nivel", "Personas": "Cantidad"},
        )
        fig.update_traces(textposition="outside")
        style_fig(fig)
        st.plotly_chart(fig, use_container_width=True)

    with colC2:
        st.metric("Total en 1–4 (predicho)", f"{int(dist_table['Personas'].sum())}")
        st.dataframe(dist_table, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.subheader("Explorar por nivel y/o departamento")
    levels_sel = st.multiselect("Nivel (1–4)", COMPANY_LEVELS_TO_SHOW, default=COMPANY_LEVELS_TO_SHOW, key="lvls_company")
    depts_sel = st.multiselect("Departamento (opcional)", sorted(df["Department"].dropna().astype(str).unique()), key="depts_company")

    mask2 = df["_level_base"].isin(levels_sel)
    if depts_sel:
        mask2 &= df["Department"].astype(str).isin(depts_sel)

    explore_cols = [id_col, "Department", "_pred_base", "_level_base"] + FEATURES
    table = df.loc[mask2, explore_cols].rename(
        columns={id_col: "ID", "_pred_base": "Predicho", "_level_base": "Nivel"}
    ).sort_values(["Nivel", "Predicho"], ascending=[True, False])

    st.dataframe(table, use_container_width=True)

    # --------- COMPARATIVA GLOBAL POR GÉNERO ---------
    st.markdown("---")
    st.subheader("👥 Comparativa de niveles por género (global compañía)")

    if gender_col is None:
        st.info("No se encontró ninguna columna de género para comparar.")
    else:
        df_gender = df[df[gender_col].notna()].copy()
        if df_gender.empty:
            st.info("No hay datos de género disponibles para comparar.")
        else:
            df_gender_levels = df_gender[df_gender["_level_base"].isin(COMPANY_LEVELS_TO_SHOW)].copy()
            dist_gender = (
                df_gender_levels
                .groupby([gender_col, "_level_base"])
                .size()
                .reset_index(name="Personas")
            )
            total_by_gender = dist_gender.groupby(gender_col)["Personas"].transform("sum")
            dist_gender["% dentro de género"] = (dist_gender["Personas"] / total_by_gender * 100).round(1)

            fig_g = px.bar(
                dist_gender,
                x="_level_base",
                y="Personas",
                color=gender_col,
                barmode="group",
                title="Distribución de niveles (1–4) por género — predicho",
                labels={
                    "_level_base": "Nivel (redondeado)",
                    "Personas": "Cantidad",
                    gender_col: "Género",
                },
                color_discrete_sequence=DEFAULT_SEQ,
            )
            fig_g.update_traces(textposition="outside")
            style_fig(fig_g)
            st.plotly_chart(fig_g, use_container_width=True)

            st.dataframe(dist_gender, use_container_width=True)

            st.markdown("##### KPIs por género (global)")
            kpi_gender = (
                df_gender
                .groupby(gender_col)["_pred_base"]
                .agg(Media_Predicho="mean", Conteo="count")
                .reset_index()
            )
            kpi_gender["Media_Predicho"] = kpi_gender["Media_Predicho"].round(3)
            st.dataframe(kpi_gender, use_container_width=True)

    # --------- COMPARATIVA GLOBAL POR AÑO DE CONTRATACIÓN ---------
    st.markdown("---")
    st.subheader("📅 Comparativa de niveles por año de contratación (DateofHire)")

    if HIRE_COL is None or df["HireYear"].notna().sum() == 0:
        st.info("No hay información de fecha de contratación suficiente para esta comparativa.")
    else:
        df_hire = df[df["HireYear"].notna() & df["_level_base"].isin(COMPANY_LEVELS_TO_SHOW)].copy()

        dist_hire = (
            df_hire
            .groupby(["HireYear", "_level_base"])
            .size()
            .reset_index(name="Personas")
        )
        total_by_year = dist_hire.groupby("HireYear")["Personas"].transform("sum")
        dist_hire["% dentro del año"] = (dist_hire["Personas"] / total_by_year * 100).round(1)

        fig_h = px.bar(
            dist_hire,
            x="HireYear",
            y="Personas",
            color="_level_base",
            barmode="stack",
            title="Distribución de niveles (1–4) por año de contratación",
            labels={
                "HireYear": "Año de contratación",
                "_level_base": "Nivel (redondeado)",
                "Personas": "Cantidad",
            },
            color_discrete_sequence=DEFAULT_SEQ,
        )
        style_fig(fig_h)
        st.plotly_chart(fig_h, use_container_width=True)

        st.dataframe(dist_hire.sort_values(["HireYear", "_level_base"]), use_container_width=True)

# =========================
# DEPARTAMENTO (con filtros)
# =========================
with tab_dept:
    st.header("Comparador global por departamento — SOLO PREDICHO")

    dept_opts = sorted(df["Department"].dropna().astype(str).unique())
    dept_sel = st.selectbox("Elige un Departamento", options=dept_opts, index=0, key="dept_sel_tab")

    df_dept = df[df["Department"].astype(str) == dept_sel].copy()
    if df_dept.empty:
        st.warning("No hay registros en el departamento seleccionado.")
    else:
        # Baseline dept
        df_dept["_pred_base"] = tree.predict(df_dept[FEATURES])
        df_dept["_level_base"] = to_level(df_dept["_pred_base"])

        # ---------- Filtros internos ----------
        st.markdown("#### Filtros del departamento")
        if gender_col is not None:
            col_f1, col_f2, col_f3, col_f4 = st.columns([1, 1, 1, 1])
        else:
            col_f1, col_f2, col_f3 = st.columns([1, 1, 1])
            col_f4 = None

        levels_pick = col_f1.multiselect("Nivel (baseline predicho)", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5], key="dept_levels")
        min_pred, max_pred = float(df_dept["_pred_base"].min()), float(df_dept["_pred_base"].max())
        pred_range = col_f2.slider(
            "Rango de PerfScoreID predicho",
            min_value=round(min_pred, 2), max_value=round(max_pred, 2),
            value=(round(min_pred, 2), round(max_pred, 2)), step=0.01, key="dept_pred_range"
        )
        search_txt = col_f3.text_input(f"Buscar por {id_col}", value="", key="dept_search").strip().lower()

        genders_pick = None
        if gender_col is not None and col_f4 is not None:
            gender_opts_dept = sorted(df_dept[gender_col].dropna().astype(str).unique())
            if gender_opts_dept:
                genders_pick = col_f4.multiselect(
                    "Género (opcional)",
                    options=gender_opts_dept,
                    default=gender_opts_dept,
                    key=f"dept_gender_{dept_sel}"
                )

        mask_dept = df_dept["_level_base"].isin(levels_pick)
        mask_dept &= (df_dept["_pred_base"].between(pred_range[0], pred_range[1], inclusive="both"))
        if search_txt:
            mask_dept &= df_dept[id_col].astype(str).str.lower().str.contains(search_txt, na=False)
        if gender_col is not None and genders_pick:
            mask_dept &= df_dept[gender_col].astype(str).isin(genders_pick)

        df_dept_filtrado = df_dept[mask_dept].copy()
        target_view = df_dept_filtrado if not df_dept_filtrado.empty else df_dept

        # ---------- KPIs del subconjunto ----------
        cA, cB, cC = st.columns(3)
        cA.metric("Personas (filtrado)", f"{len(target_view)}")
        cB.metric("PerfScoreID medio (predicho)", f"{target_view['_pred_base'].mean():.3f}")
        cC.metric("Nivel más frecuente", f"{target_view['_level_base'].mode().iloc[0] if not target_view.empty else df_dept['_level_base'].mode().iloc[0]}")

        # ---------- Distribución ----------
        dist_base = target_view["_level_base"].value_counts().sort_index()
        dist_table = pd.DataFrame({
            "Nivel": dist_base.index,
            "Personas": dist_base.values,
            "Porcentaje": (dist_base.values / max(1, len(target_view)) * 100).round(1)
        })

        fig_dist = px.bar(
            dist_table, x="Nivel", y="Personas", text="Personas",
            title=f"Distribución de niveles — Baseline (predicho) — {dept_sel} (filtro aplicado)",
            labels={"Nivel": "Nivel (redondeado)", "Personas": "Cantidad"},
        )
        fig_dist.update_traces(textposition="outside")
        style_fig(fig_dist)
        st.plotly_chart(fig_dist, use_container_width=True)

        st.dataframe(
            target_view[[id_col, "_pred_base", "_level_base"] + FEATURES].rename(
                columns={id_col: "ID", "_pred_base": "Predicho", "_level_base": "Nivel"}
            ).sort_values(["Nivel", "Predicho"], ascending=[True, False]),
            use_container_width=True
        )

        # ---------- Comparativa por género dentro del departamento ----------
        st.markdown("##### Comparativa de niveles por género en el departamento")
        if gender_col is None:
            st.info("No se encontró columna de género en el dataset, no se puede mostrar la comparativa por género.")
        else:
            df_dept_gender = target_view[target_view[gender_col].notna()].copy()
            if df_dept_gender.empty:
                st.info("No hay datos de género en el subconjunto seleccionado.")
            else:
                dist_dept_gender = (
                    df_dept_gender
                    .groupby([gender_col, "_level_base"])
                    .size()
                    .reset_index(name="Personas")
                )
                total_by_gender_dept = dist_dept_gender.groupby(gender_col)["Personas"].transform("sum")
                dist_dept_gender["% dentro de género"] = (dist_dept_gender["Personas"] / total_by_gender_dept * 100).round(1)

                fig_dept_gender = px.bar(
                    dist_dept_gender,
                    x="_level_base",
                    y="Personas",
                    color=gender_col,
                    barmode="group",
                    title=f"Distribución de niveles por género — {dept_sel}",
                    labels={
                        "_level_base": "Nivel (redondeado)",
                        "Personas": "Cantidad",
                        gender_col: "Género",
                    },
                    color_discrete_sequence=DEFAULT_SEQ,
                )
                fig_dept_gender.update_traces(textposition="outside")
                style_fig(fig_dept_gender)
                st.plotly_chart(fig_dept_gender, use_container_width=True)

                st.dataframe(dist_dept_gender, use_container_width=True)

        # ---------- Comparativa por año de contratación en el departamento ----------
        st.markdown("##### Comparativa de niveles por año de contratación en el departamento")
        if HIRE_COL is None or target_view["HireYear"].notna().sum() == 0:
            st.info("No hay información de fecha de contratación suficiente en este departamento.")
        else:
            df_dept_hire = target_view[target_view["HireYear"].notna()].copy()
            dist_dept_hire = (
                df_dept_hire
                .groupby(["HireYear", "_level_base"])
                .size()
                .reset_index(name="Personas")
            )
            total_by_year_dept = dist_dept_hire.groupby("HireYear")["Personas"].transform("sum")
            dist_dept_hire["% dentro del año"] = (dist_dept_hire["Personas"] / total_by_year_dept * 100).round(1)

            fig_dept_hire = px.bar(
                dist_dept_hire,
                x="HireYear",
                y="Personas",
                color="_level_base",
                barmode="stack",
                title=f"Niveles por año de contratación — {dept_sel}",
                labels={
                    "HireYear": "Año de contratación",
                    "_level_base": "Nivel (redondeado)",
                    "Personas": "Cantidad",
                },
                color_discrete_sequence=DEFAULT_SEQ,
            )
            style_fig(fig_dept_hire)
            st.plotly_chart(fig_dept_hire, use_container_width=True)

            st.dataframe(dist_dept_hire.sort_values(["HireYear", "_level_base"]), use_container_width=True)

        # ---------- Escenario con palancas ----------
        st.markdown("##### Escenario con palancas")
        apply_on_subset = st.checkbox("Aplicar palancas solo al subconjunto filtrado", value=True, key="apply_subset")

        comp_levels = None
        fig_comp = None

        if st.button("Aplicar palancas y comparar", use_container_width=True, key="apply_dept"):
            base_set = target_view if apply_on_subset else df_dept

            base_set = base_set.copy()
            base_set["_pred_base"] = tree.predict(base_set[FEATURES])
            base_set["_level_base"] = to_level(base_set["_pred_base"])

            dept_scn = apply_levers(base_set)
            dept_scn["_pred_scn"] = tree.predict(dept_scn[FEATURES])
            dept_scn["_level_scn"] = to_level(dept_scn["_pred_scn"])

            k1, k2, k3 = st.columns(3)
            mean_base = base_set["_pred_base"].mean()
            mean_scn = dept_scn["_pred_scn"].mean()
            uplift_abs = (dept_scn["_pred_scn"] - base_set["_pred_base"]).mean()
            uplift_pct = 100 * (mean_scn / mean_base - 1) if mean_base != 0 else 0.0

            k1.metric("Media baseline", f"{mean_base:.3f}")
            k2.metric("Media escenario", f"{mean_scn:.3f}", delta=f"{mean_scn - mean_base:+.3f} ({uplift_pct:+.1f}%)")
            k3.metric("Uplift medio", f"{uplift_abs:+.3f}")

            dist_base2 = base_set["_level_base"].value_counts().sort_index()
            dist_scn = dept_scn["_level_scn"].value_counts().sort_index()
            comp_levels = pd.DataFrame({"Nivel": sorted(set(dist_base2.index).union(set(dist_scn.index)))})
            comp_levels["Personas_baseline"] = comp_levels["Nivel"].map(dist_base2).fillna(0).astype(int)
            comp_levels["Personas_escenario"] = comp_levels["Nivel"].map(dist_scn).fillna(0).astype(int)

            fig_comp = px.bar(
                comp_levels.melt(id_vars="Nivel", value_vars=["Personas_baseline", "Personas_escenario"],
                                 var_name="Serie", value_name="Personas"),
                x="Nivel", y="Personas", color="Serie", barmode="group",
                title=f"Número de personas por nivel — Baseline vs. Escenario (predicho) — {dept_sel}" + (" (subconjunto)" if apply_on_subset else " (todo el área)"),
                labels={"Nivel": "Nivel", "Personas": "Cantidad"},
                color_discrete_sequence=DEFAULT_SEQ
            )
            style_fig(fig_comp)

            st.plotly_chart(fig_comp, use_container_width=True)
            st.dataframe(comp_levels, use_container_width=True, hide_index=True)

            # Descarga comparador CSV
            down = dept_scn.copy()
            down["Pred_baseline"] = base_set["_pred_base"].values
            down["Nivel_baseline"] = base_set["_level_base"].values
            down["Pred_escenario"] = dept_scn["_pred_scn"].values
            down["Nivel_escenario"] = dept_scn["_level_scn"].values
            st.download_button(
                "Descargar comparación (CSV)",
                down.to_csv(index=False),
                "comparacion_departamento.csv",
                use_container_width=True
            )

            st.session_state[f"comp_levels_{dept_sel}"] = comp_levels
            st.session_state[f"fig_comp_{dept_sel}"] = fig_comp

        # ---------- Notas + Informe ----------
        st.markdown("---")
        st.subheader("📝 Notas del departamento (ideas y sugerencias)")
        saved = get_dept_notes(dept_sel)
        ideas = st.text_area("Ideas de mejora a trabajar", value=saved["ideas"], key=f"ideas_{dept_sel}", height=120, placeholder="Acciones a priorizar, quick wins, fricciones, etc.")
        sugerencias = st.text_area("Sugerencias de mejora", value=saved["sugerencias"], key=f"sugs_{dept_sel}", height=120, placeholder="Intervenciones propuestas, cambios de proceso, formaciones, etc.")
        colN1, colN2 = st.columns([1,1])
        if colN1.button("💾 Guardar notas", key=f"save_notes_{dept_sel}"):
            set_dept_notes(dept_sel, ideas, sugerencias)
            st.success("Notas guardadas.")

        st.markdown("#### 📄 Generar informe HTML del departamento")
        include_scenario = st.checkbox("Incluir escenario con palancas (si existe)", value=True)

        kpis_dict = {
            "personas": len(target_view),
            "media": target_view["_pred_base"].mean(),
            "moda": int(target_view["_level_base"].mode().iloc[0]) if not target_view.empty else int(df_dept["_level_base"].mode().iloc[0]),
        }
        fig_dist_for_html = fig_dist
        comp_tbl_for_html = st.session_state.get(f"comp_levels_{dept_sel}") if include_scenario else None
        fig_comp_for_html = st.session_state.get(f"fig_comp_{dept_sel}") if include_scenario else None

        ideas_for_html = (ideas or "").replace("\n", "<br>")
        sugerencias_for_html = (sugerencias or "").replace("\n", "<br>")

        html_str = build_report_html(
            dept_sel,
            dist_table,
            fig_dist_for_html,
            comp_tbl_for_html,
            fig_comp_for_html,
            ideas_for_html,
            sugerencias_for_html,
            kpis_dict,
        )

        st.download_button(
            "⬇️ Descargar informe HTML del departamento",
            data=html_str.encode("utf-8"),
            file_name=f"informe_{dept_sel.replace(' ','_')}.html",
            mime="text/html",
            use_container_width=True
        )

# =========================
# INDIVIDUAL
# =========================
with tab_individual:
    st.header("Predicción y what-if individual — SOLO PREDICHO")
    colF1, colF2 = st.columns([1, 1])
    dept_filter = colF1.selectbox("Filtra por Departamento", options=["(Todos)"] + sorted(df["Department"].dropna().astype(str).unique()))
    level_filter = colF2.multiselect("Filtra por Nivel (baseline predicho)", [1, 2, 3, 4, 5], default=[1, 2, 3, 4, 5])

    mask_ind = df["_level_base"].isin(level_filter)
    if dept_filter != "(Todos)":
        mask_ind &= (df["Department"].astype(str) == dept_filter)

    base_cols = [id_col, "Department", "_pred_base", "_level_base"] + FEATURES
    if gender_col is not None:
        base_cols.append(gender_col)
    if HIRE_COL is not None:
        base_cols.extend(["DateofHire_dt", "TenureYears"])

    candidates = df.loc[mask_ind, base_cols].copy()
    rename_map = {id_col: "ID", "_pred_base": "Predicho", "_level_base": "Nivel", "DateofHire_dt": "DateofHire", "TenureYears": "TenureYears"}
    if gender_col is not None:
        rename_map[gender_col] = "Género"
    candidates = candidates.rename(columns=rename_map)
    st.dataframe(candidates.sort_values(["Nivel", "Predicho"], ascending=[True, False]), use_container_width=True)

    emp_opts = df.loc[mask_ind, id_col].astype(str).tolist()
    if not emp_opts:
        st.info("No hay personas con ese filtro.")
    else:
        emp_sel = st.selectbox("Selecciona ID", options=emp_opts, key="emp_select")

        row = df[df[id_col].astype(str) == emp_sel].iloc[0:1].copy()
        row["_pred_base"] = tree.predict(row[FEATURES])
        row["_level_base"] = to_level(row["_pred_base"])

        hire_dt_value = row["DateofHire_dt"].iloc[0] if "DateofHire_dt" in row.columns else pd.NaT
        tenure_value = row["TenureYears"].iloc[0] if "TenureYears" in row.columns else np.nan

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Departamento", f"{row['Department'].iloc[0]}")
        c2.metric("PerfScoreID predicho (baseline)", f"{float(row['_pred_base'].iloc[0]):.3f}")
        c3.metric("Nivel (baseline)", f"{int(row['_level_base'].iloc[0])}")
        if pd.notna(hire_dt_value):
            c4.metric("Fecha de contratación", hire_dt_value.strftime("%Y-%m-%d"))
        else:
            c4.metric("Fecha de contratación", "sin datos")

        # ---------- Contexto por género y antigüedad ----------
        if gender_col is not None:
            gender_val = str(row[gender_col].iloc[0]) if pd.notna(row[gender_col].iloc[0]) else None
        else:
            gender_val = None

        if gender_val is not None:
            st.markdown("##### Contexto de la persona dentro de su género y departamento")
            dept_name = str(row["Department"].iloc[0])
            mask_dept_all = df["Department"].astype(str) == dept_name
            dept_all = df[mask_dept_all].copy()
            dept_all["_pred_base"] = tree.predict(dept_all[FEATURES])

            mask_dept_gender = mask_dept_all & (df[gender_col].astype(str) == gender_val)
            dept_gender = df[mask_dept_gender].copy()
            if not dept_gender.empty:
                dept_gender["_pred_base"] = tree.predict(dept_gender[FEATURES])

            colG1, colG2, colG3, colG4 = st.columns(4)
            colG1.metric("Género de la persona", gender_val)
            colG2.metric("Media depto (todos)", f"{dept_all['_pred_base'].mean():.3f}")
            if not dept_gender.empty:
                colG3.metric(f"Media depto ({gender_val})", f"{dept_gender['_pred_base'].mean():.3f}")
            else:
                colG3.metric(f"Media depto ({gender_val})", "s/datos")

            if pd.notna(tenure_value):
                # Media de TenureYears en el departamento
                if "TenureYears" in dept_all.columns:
                    colG4.metric("Antigüedad persona vs depto",
                                 f"{tenure_value:.1f} años",
                                 delta=f"{tenure_value - dept_all['TenureYears'].mean():+.1f} años")
                else:
                    colG4.metric("Antigüedad persona", f"{tenure_value:.1f} años")
            else:
                colG4.metric("Antigüedad persona", "sin datos")
        else:
            if gender_col is not None:
                st.info("La persona seleccionada no tiene información de género, no se muestra el contexto por género.")

        st.markdown("##### Palancas individuales (deltas absolutos)")
        colA, colB, colC = st.columns(3)
        i_satisfaction = colA.number_input("EmpSatisfaction (Δ)", value=0.0, step=0.1, min_value=-10.0, max_value=10.0)
        i_engagement = colB.number_input("EngagementSurvey (Δ)", value=0.0, step=0.05, min_value=-10.0, max_value=10.0)
        i_salary_pct = colC.number_input("Salary (%)", value=0.0, step=1.0, min_value=-100.0, max_value=100.0)

        colD, colE, colF = st.columns(3)
        i_projects = colD.number_input("SpecialProjectsCount (Δ)", value=0.0, step=0.5, min_value=-50.0, max_value=50.0)
        i_absences = colE.number_input("Absences (Δ)", value=0.0, step=0.5, min_value=-365.0, max_value=365.0)
        i_lates = colF.number_input("DaysLateLast30 (Δ)", value=0.0, step=0.5, min_value=-30.0, max_value=30.0)

        if st.button("Calcular escenario individual", use_container_width=True):
            row_scn = row.copy()
            row_scn.loc[:, "EmpSatisfaction"]      = float(row_scn["EmpSatisfaction"])      + i_satisfaction
            row_scn.loc[:, "EngagementSurvey"]     = float(row_scn["EngagementSurvey"])     + i_engagement
            row_scn.loc[:, "Salary"]               = float(row_scn["Salary"])               * (1 + i_salary_pct/100.0)
            row_scn.loc[:, "SpecialProjectsCount"] = float(row_scn["SpecialProjectsCount"]) + i_projects
            row_scn.loc[:, "Absences"]             = float(row_scn["Absences"])             + i_absences
            row_scn.loc[:, "DaysLateLast30"]       = float(row_scn["DaysLateLast30"])       + i_lates

            for c in ["EmpSatisfaction","EngagementSurvey","SpecialProjectsCount","Absences","DaysLateLast30","Salary"]:
                row_scn.loc[:, c] = max(0.0, float(row_scn[c]))

            row_scn["_pred_scn"]  = tree.predict(row_scn[FEATURES])
            row_scn["_level_scn"] = to_level(row_scn["_pred_scn"])

            kA, kB, kC = st.columns(3)
            base = float(row["_pred_base"].iloc[0])
            esc  = float(row_scn["_pred_scn"].iloc[0])
            delta_abs = esc - base
            delta_pct = 100 * (esc / base - 1) if base != 0 else 0.0

            kA.metric("Predicho (baseline)", f"{base:.3f}")
            kB.metric("Predicho (escenario)", f"{esc:.3f}", delta=f"{delta_abs:+.3f} ({delta_pct:+.1f}%)")
            kC.metric("Cambio de nivel", f"{int(row_scn['_level_scn'].iloc[0]) - int(row['_level_base'].iloc[0]):+d}")

            before = row[FEATURES].T.rename(columns={row.index[0]: "Baseline"})
            after = row_scn[FEATURES].T.rename(columns={row_scn.index[0]: "Escenario"})
            comp = before.join(after)
            st.write("**Variables (baseline vs escenario)**")
            st.dataframe(comp, use_container_width=True)

# =========================
# NOTAS FINALES
# =========================
st.caption(
    "• El tablero usa SOLO predicción del modelo (árbol sencillo) sobre EmpSatisfaction, EngagementSurvey, Salary, "
    "SpecialProjectsCount, Absences y DaysLateLast30. "
    "• Se han añadido comparativas por género (usando la columna 'Sex') y por año de contratación (DateofHire), "
    "además de un contexto de antigüedad aproximada (TenureYears)."
)
