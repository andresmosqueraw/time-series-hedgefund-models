# Hedge Fund Time Series Forecasting

**Universidad de los Andes, Bogotá, Colombia**

Proyecto final del curso de Machine Learning para Series de Tiempo. Solución a la competencia de Kaggle [Hedge Fund Time Series Forecasting](https://www.kaggle.com/competitions/ts-forecasting), que consiste en predecir `y_target` para ~1.4 millones de filas de datos financieros completamente anonimizados, maximizando el **Weighted RMSE Skill Score**.

**Score final en el leaderboard público: 0.3429** (objetivo de la competencia: > 0.30)

---

## Tabla de contenidos

1. [Contexto del problema](#contexto-del-problema)
2. [Estructura del repositorio](#estructura-del-repositorio)
3. [Requisitos e instalación](#requisitos-e-instalación)
4. [Dashboard Streamlit](#dashboard-streamlit)
5. [Experimentos](#experimentos)
6. [Inferencia con el modelo publicado](#inferencia-con-el-modelo-publicado)
7. [Paper](#paper)

---

## Contexto del problema

Un fondo de cobertura (Hedge Fund) activo aporta datos financieros con:

- **5.3 M filas** de entrenamiento (`train.parquet`) y **1.4 M** de test (`test.parquet`)
- **86 features** numéricas anónimas (`feature_a` … `feature_ch`)
- 4 identificadores jerárquicos: `code`, `sub_code`, `sub_category`, `horizon`
- Eje temporal entero `ts_index` (1 → 4 376); el test ocupa `ts_index > 3 601`
- **Restricción estricta:** ningún estadístico derivado puede referenciar `ts_index > t` — usar información futura invalida la submission

La métrica de evaluación es:

```
Score = sqrt(1 - clip(Σ w·(y - ŷ)² / Σ w·y², 0, 1))
```

Un score de 0 equivale a predecir siempre cero; 1 es predicción perfecta.

---

## Estructura del repositorio

```
oficial-entrega/
│
├── README.md                        ← este archivo
├── main-problem.md                  ← definición compacta del problema
│
├── experimentos/                    ← todos los experimentos (ver sección dedicada)
│   ├── mlp.py / mlp.ipynb
│   ├── lstm.py / lstm.ipynb
│   ├── score-0-2675-*.py / *.ipynb
│   ├── score-0-2712-*.py / *.ipynb
│   ├── score-0-2824-*.py / *.ipynb
│   ├── score-0-3212-*.py / *.ipynb
│   ├── score-0-3377-*.py / *.ipynb
│   ├── score-0-3429-*.py / *.ipynb
│   └── inference-example-ensemble-dual.py
│
├── streamlit/
│   ├── app.py                       ← dashboard interactivo
│   └── requirements.txt
│
├── paper/
│   └── time_series_forecasting/
│       ├── neurips_2025.tex         ← paper principal
│       ├── neurips_2025.sty
│       └── references.bib
│
└── datasets/                        ← datos de Kaggle (no incluidos en el repo)
```

---

## Requisitos e instalación

### Dependencias del dashboard Streamlit

```bash
pip install streamlit plotly pandas numpy
```

### Dependencias de los experimentos

Cada experimento indica sus dependencias al inicio. El conjunto completo es:

```bash
pip install pandas numpy lightgbm xgboost scikit-learn polars joblib tqdm torch huggingface_hub
```

> Los experimentos se diseñaron y ejecutaron en **Kaggle Notebooks** (Python 3.10, CPU). Para correrlos localmente necesitas los archivos `train.parquet` y `test.parquet` de la competencia y ajustar las rutas de los archivos (ver sección de cada experimento).

### Versiones de referencia (entorno Kaggle)

| Paquete | Versión |
|---|---|
| Python | 3.10.x |
| pandas | 2.0.3 |
| numpy | 1.24.3 |
| lightgbm | 4.1.0 |
| scikit-learn | 1.3.x |

---

## Dashboard Streamlit

El dashboard presenta de forma visual e interactiva todos los resultados del proyecto.

### Cómo ejecutarlo

```bash
cd streamlit/
streamlit run app.py
```

Se abrirá automáticamente en `http://localhost:8501`.

### Páginas disponibles

| Página | Contenido |
|---|---|
| **El Problema** | Descripción del reto, la métrica y la línea de tiempo del `ts_index` |
| **Los Datos** | Diccionario de columnas, estadísticas de `y_target` por horizonte, gráfico de curtosis vs. score, indicador de SNR |
| **Experimentos** | Gráfico de barras con la evolución del score en los 8 experimentos; tabs con detalle de cada uno y las 3 lecciones clave |
| **Demo: Ensemble Dual** | Simulador interactivo del blending por potencia — ajusta horizonte, sub-categoría, predicciones de cada pool y potencia de mezcla; incluye gráfico de pesos vs. potencia y mapa de calor |
| **Métrica Interactiva** | Calculadora manual del Weighted RMSE Skill Score con tabla comparativa de todos los experimentos |

> El dashboard no requiere los datasets de Kaggle; trabaja exclusivamente con los números del paper.

---

## Experimentos

Los archivos en `experimentos/` están nombrados con el score de leaderboard que obtuvieron. Todos los scripts `.py` son versiones exportadas de los notebooks `.ipynb` correspondientes y contienen el mismo código.

### Rutas de datos

Todos los experimentos apuntan por defecto a las rutas de Kaggle:

```python
TRAIN_PATH = '/kaggle/input/competitions/ts-forecasting/train.parquet'
TEST_PATH  = '/kaggle/input/competitions/ts-forecasting/test.parquet'
```

Para ejecutarlos **localmente** descarga los datos de la competencia y cambia esas variables al inicio de cada script.

---

### Resumen cronológico de experimentos

| Archivo | Score | Descripción |
|---|---|---|
| `mlp.py` | 0.0634 | Red neuronal densa con embeddings de entidad |
| `lstm.py` | 0.0891 | Red recurrente (LSTM) con embeddings de entidad |
| `score-0-2675-lightgbm-multi-horizonte-con-ingenieria-features-causales.py` | 0.2675 | LightGBM multi-horizonte, 183 features causales, 5 semillas |
| `score-0-2712-lightbm-con-polars-suavizado-outliers-seleccion-features.py` | 0.2712 | LightGBM + Polars + filtro Hampel + selección de features en 2 pasadas |
| `score-0-2824-stacking-lightbm-xgboost-ridge-pesos-recencia.py` | 0.2824 | Stacking LGB + XGB + Ridge con ponderación por recencia |
| `score-0-3212-lightbm-minimalista-ids-jerarquicos-como-features.py` | 0.3212 | LightGBM minimalista — IDs jerárquicos como `category` |
| `score-0-3377-ligthbm-minimalista-umbral-validacion-ajustado-entrenamiento-determinista.py` | 0.3377 | Igual al anterior + umbral de validación 3503 + `deterministic=True` |
| `score-0-3429-ensemble-dual-lightgbm-modelo-horizonte-sub-categoria-mezcla-ponderada-potencia.py` | **0.3429** | **Mejor modelo:** ensemble dual 4 modelos×horizonte + 5 modelos×sub-cat, mezcla p^2.5 |

---

### Experimento MLP (`mlp.py` / `mlp.ipynb`)

Red neuronal densa con embeddings de entidad. Submuestreo del 25 % de entidades para reducir tiempo de entrenamiento.

**Cómo ejecutar:**

```bash
# En Kaggle Notebook (recomendado) o localmente con GPU/CPU:
python experimentos/mlp.py
```

**Parámetros clave:**

- Arquitectura: `[2048, 1024, 512, 256, 128, 64]` con BatchNorm, ReLU, Dropout(0.3)
- Optimizador: Adam (lr=1e-3), scheduler ReduceLROnPlateau
- Épocas: 50 (reentrenamiento final con CosineAnnealingLR)
- Batch: 4096

**Por qué falló (score 0.0634):** el submuestreo al 25 % reduce la señal disponible, la curtosis extrema del target (> 500 en h=1) domina la función MSE con outliers y la red carece de las transformaciones temporales causales que aprovecha LightGBM.

---

### Experimento LSTM (`lstm.py` / `lstm.ipynb`)

Red recurrente con embeddings de entidad. Mismo submuestreo del 25 %.

**Cómo ejecutar:**

```bash
python experimentos/lstm.py
```

**Parámetros clave:**

- Arquitectura: `LSTM(256) → Dense(128, relu) → Dropout(0.2) → Dense(64, relu) → Dense(1)`
- Embeddings de dim 86 para cada variable categórica
- Optimizador: Adam (lr=1e-5), scheduler ReduceLROnPlateau (factor=0.5, paciencia=10)
- Función de pérdida: MAE | Épocas: 100 | Batch: 256

---

### Experimento 1 — LightGBM multi-horizonte con features causales (`score-0-2675-*.py`)

Un modelo LightGBM independiente por horizonte (h ∈ {1, 3, 10, 25}), con 183 estadísticos causales construidos en 6 grupos: identidad, derivadas temporales, normalización cross-seccional, momentum, estacionalidad y rolling/rezagos.

**Cómo ejecutar:**

```bash
python experimentos/score-0-2675-lightgbm-multi-horizonte-con-ingenieria-features-causales.py
```

**Parámetros clave:**

- `num_leaves`: 70–90 por horizonte | `lambda_l2`: 8–12 | `min_child_samples`: 150–250
- `feature_fraction`: 0.6 | `bagging_fraction`: 0.7
- Ensemble de 5 semillas: `[42, 2024, 777, 1337, 9999]`
- Corte de validación: `ts_index > 3500`

**Resultado local por horizonte:**

| Horizonte | Score local |
|---|---|
| h=1 | 0.07547 |
| h=3 | 0.13232 |
| h=10 | 0.23661 |
| h=25 | 0.29283 |
| **Global** | **0.25083** |

---

### Experimento 3 — LightGBM + Polars + filtro Hampel (`score-0-2712-*.py`)

Pipeline reimplementado en Polars para reducir uso de memoria, con filtro Pseudo-Hampel para suavizado de outliers y selección de features en dos pasadas.

**Cómo ejecutar:**

```bash
pip install polars
python experimentos/score-0-2712-lightbm-con-polars-suavizado-outliers-seleccion-features.py
```

**Innovaciones:**

- Reducción automática de tipos (Float64→Float32, Int64→Int8/16/32)
- Filtro Hampel: mediana rolling w=50, umbral 5σ sobre 12 features ruidosas
- Features adicionales: sign×magnitud, combinaciones aritméticas entre pares, rolling robusto (w=1000), EMA/MACD
- Selección de features más importantes por horizonte: N₁=197, N₃=177, N₁₀=197, N₂₅=217
- 5 semillas con entrenamiento paralelo (`joblib`, 2 workers)

---

### Experimento 4 — Stacking LGB + XGB + Ridge (`score-0-2824-*.py`)

Stacking de LightGBM y XGBoost con meta-modelo Ridge y ponderación por recencia de hasta 40.5× para las observaciones más recientes.

**Cómo ejecutar:**

```bash
pip install xgboost
python experimentos/score-0-2824-stacking-lightbm-xgboost-ridge-pesos-recencia.py
```

**Innovaciones:**

- Pesos por recencia: `w̃ᵢ = wᵢ · (0.5 + 40.0 · t̂ᵢ)` donde `t̂ᵢ` es el tiempo normalizado
- Meta-modelo Ridge (α=1, pesos no negativos, 5-Fold sin barajado) sobre predicciones OOF
- Features adicionales: wavelets EMA, indicador de régimen triple, oscilador estocástico, z-scores de reversión, desvíos cross-seccionales

**Pesos Ridge resultantes por horizonte:**

| h | w_LGB | w_XGB | Score stacking |
|---|---|---|---|
| 1 | 78.6 % | 21.4 % | 0.081 |
| 3 | 64.4 % | 35.6 % | 0.136 |
| 10 | 94.3 % | 5.7 % | 0.244 |
| 25 | 90.6 % | 9.4 % | 0.278 |

---

### Experimento 5 — LightGBM minimalista con IDs jerárquicos (`score-0-3212-*.py`)

El mayor salto del proyecto (+0.04 sobre el experimento anterior). Descubrimiento clave: pasar `code`, `sub_code`, `sub_category` y `horizon` directamente como `dtype='category'` permite a LightGBM aprender efectos fijos implícitos por entidad.

**Cómo ejecutar:**

```bash
python experimentos/score-0-3212-lightbm-minimalista-ids-jerarquicos-como-features.py
```

**Parámetros clave:**

```python
lgb_cfg = {
    'learning_rate': 0.035,
    'n_estimators': 1202,
    'num_leaves': 64,
    'min_child_samples': 100,
    'feature_fraction': 0.75,
}
VAL_THRESHOLD = 3500
```

**Convergencia (iteraciones de early stopping):**

| Horizonte | Mejor iteración | RMSE validación |
|---|---|---|
| h=1 | 133 | 0.001134 |
| h=3 | 93 | 0.001818 |
| h=10 | 107 | 0.002676 |
| h=25 | 117 | 0.002908 |

---

### Experimento 6 — LightGBM minimalista determinista (`score-0-3377-*.py`)

Mismo pipeline del Experimento 5 con tres ajustes: desplazamiento del umbral de validación (+3 observaciones), tasa de aprendizaje reducida y entrenamiento 100 % reproducible.

**Cómo ejecutar:**

```bash
python experimentos/score-0-3377-ligthbm-minimalista-umbral-validacion-ajustado-entrenamiento-determinista.py
```

**Cambios respecto al Experimento 5:**

```python
VAL_THRESHOLD = 3503   # era 3500
'learning_rate': 0.030  # era 0.035
'feature_fraction': 0.70  # era 0.75
'seed': 5
'deterministic': True
'force_row_wise': True
'num_threads': 1
```

> Nota: el desplazamiento de solo 3 observaciones (0.08 % del eje temporal) produjo +0.0165 en el score público, lo que ilustra la alta sensibilidad al esquema de validación cuando el SNR es extremadamente bajo.

---

### Experimento 7 — Ensemble dual LightGBM (mejor modelo, `score-0-3429-*.py`)

El experimento ganador. Entrena **9 modelos en total** y fusiona sus predicciones mediante mezcla ponderada por potencia:

- **Pool A:** 4 modelos, uno por horizonte (h ∈ {1, 3, 10, 25}) — especialización temporal
- **Pool B:** 5 modelos, uno por sub-categoría — especialización estructural

La mezcla final para cada fila usa la fórmula:

```
ŷ = (score_h^p / (score_h^p + score_s^p)) · ŷ_h
  + (score_s^p / (score_h^p + score_s^p)) · ŷ_s
```

donde `p = 2.5` amplifica la diferencia entre modelos buenos y débiles.

**Cómo ejecutar:**

```bash
python experimentos/score-0-3429-ensemble-dual-lightgbm-modelo-horizonte-sub-categoria-mezcla-ponderada-potencia.py
```

**Parámetros clave:**

```python
BLEND_POWER = 2.5
VAL_THRESHOLD = 3500

LGB_PARAMS = {
    'learning_rate': 0.03,
    'n_estimators': 1200,
    'num_leaves': 64,
    'min_child_samples': 100,
    'feature_fraction': 0.75,
    'bagging_fraction': 0.75,
    'bagging_freq': 5,
    'random_state': 101,
}
```

**Scores de validación local por modelo:**

| Pool | Modelo | Score local |
|---|---|---|
| Horizonte | h=1 | 0.0731 |
| Horizonte | h=3 | 0.1289 |
| Horizonte | h=10 | 0.2510 |
| Horizonte | h=25 | 0.3271 |
| Sub-cat | DPPUO5X2 | 0.2126 |
| Sub-cat | NQ58FVQM | 0.2367 |
| Sub-cat | PHHHVYZI | 0.2879 |
| Sub-cat | PZ9S1Z4V | 0.3133 |
| Sub-cat | V8BKY1IV | 0.2203 |

Peso promedio resultante: **67.1 % Pool B (sub-categoría)**, 32.9 % Pool A (horizonte).

**Tiempo de entrenamiento estimado:** ~20–25 minutos en Kaggle CPU (no se recomienda GPU).

**Salida:** `submission.csv` con columnas `id` y `prediction` (1 447 107 filas).

---

### Inferencia con modelo pre-entrenado (`inference-example-ensemble-dual.py`)

El pipeline del Experimento 7 está publicado en Hugging Face Hub. Este script permite hacer predicciones sin reentrenar.

**Cómo ejecutar:**

```bash
pip install lightgbm joblib pandas numpy huggingface_hub

python experimentos/inference-example-ensemble-dual.py
```

El script:
1. Descarga automáticamente `full_pipeline.pkl` desde `andrewmos/lightbm-ts-forecasting-kaggle` en Hugging Face
2. Extrae los modelos de horizonte, modelos de sub-categoría, estadísticos de target encoding y scores de blending
3. Aplica la misma ingeniería de features usada en entrenamiento
4. Genera `predictions.csv`

**Requisito:** ajusta la ruta del dataset de test en la línea:

```python
test_df = pd.read_csv("test.csv")   # ← cambia por tu ruta
```

---

## Paper

El paper completo en formato NeurIPS 2025 se encuentra en `paper/time_series_forecasting/neurips_2025.tex`.

**Para compilar el PDF:**

```bash
cd paper/time_series_forecasting/
pdflatex neurips_2025.tex
bibtex neurips_2025
pdflatex neurips_2025.tex
pdflatex neurips_2025.tex
```

Requiere una instalación de LaTeX (TeX Live, MiKTeX o similar) con los paquetes `booktabs`, `amsmath`, `babel` (spanish), `hyperref` y `microtype`.

El paper cubre: introducción y justificación, tres trabajos relacionados (TimeXer, CATS, LightGBM), descripción del dataset, metodología de los 8 experimentos y discusión de resultados.

---

## Lecciones aprendidas

1. **LightGBM domina a las redes neuronales** en datos tabulares financieros de bajo SNR — margen de 3×–4× en score.
2. **La identidad de la serie importa más que las features:** pasar los identificadores jerárquicos como `dtype='category'` fue el mayor salto individual del proyecto (+0.04 en score público), superando meses de ingeniería de features.
3. **Más complejidad no implica mejor generalización:** el modelo minimalista (Exp. 5) superó al stacking LGB+XGB+Ridge (Exp. 4) por +0.04.
4. **El esquema de validación es crítico:** un desplazamiento de apenas 3 observaciones en el umbral de validación (ts_index 3500 → 3503) produjo +0.0165 en el leaderboard público.
