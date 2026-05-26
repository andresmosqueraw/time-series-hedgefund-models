# Guión: Presentación "No Free Lunch — Predicción de Series de Tiempo de Hedge Fund"
**Duración total:** 15 minutos | **Diapositivas:** 10 | **Equipo:** Arthur XXIV

---

## Diapositiva 1 — Portada
**⏱ ~30 segundos**

### Qué poner:
- Título: **"¿Puede una máquina adivinar el futuro del dinero?"**
- Subtítulo pequeño: *Hedge Fund Time Series Forecasting — Kaggle Competition*
- Nombres del equipo y universidad
- Logo de Kaggle o una imagen de una gráfica financiera subiendo y bajando

### Qué decir:
> "Hola a todos. Hoy les vamos a contar cómo le enseñamos a una computadora a predecir números financieros del futuro — usando solo datos del pasado. Y veremos si lo logramos."

---

## Diapositiva 2 — El Problema
**⏱ ~2 minutos**

### Qué poner:
- **Analogía visual:** imagen de un niño intentando adivinar cuántos dulces habrá mañana en un frasco que cambia cada día
- Texto simple a la izquierda:
  - "Un fondo de inversión (Hedge Fund) maneja **más de $4 billones de dólares**"
  - "Necesita predecir valores financieros **del futuro** usando solo datos **del pasado**"
  - "Los datos son **anónimos** — no sabemos qué son exactamente"
- A la derecha, la métrica de la competencia:

```
Score = √(1 − error² / varianza_real)
Meta: Score > 0.30
```

- **Regla de oro en rojo:** "⛔ Solo puedes usar datos del pasado. Nada del futuro."

### Qué decir:
> "Imaginen que tienen un frasco con dulces y cada día alguien agrega o saca algunos. Su tarea es adivinar cuántos habrá mañana, pero **solo mirando lo que pasó antes** — nunca lo que pasa después. Eso es exactamente lo que hicimos, pero con datos financieros reales de un fondo que maneja billones de dólares. La competencia en Kaggle nos da un 'score' entre 0 y 1 — mientras más cerca de 1, mejor predecimos."

---

## Diapositiva 3 — Los Datos
**⏱ ~1.5 minutos**

### Qué poner:
- Tabla simple y visual (como una hoja de excel resumida):

| ¿Qué es? | En el dataset | Ejemplo |
|---|---|---|
| ¿Quién es? | `code`, `sub_code` | Entidad A, Subgrupo B |
| ¿Cuándo? | `ts_index` (número entero) | 1, 2, 3 … 4376 |
| ¿A qué plazo? | `horizon` | 1, 3, 10 o 25 pasos |
| ¿Cuánto vale? | `y_target` (solo en train) | −158 a +87 |
| Pistas | `feature_a` … `feature_ch` | 86 columnas misteriosas |

- Números clave en grande:
  - **5.3 millones** de filas de entrenamiento
  - **1.4 millones** de filas a predecir
  - **86 features** anónimas
  - Señal-ruido bajísimo: **SNR ≈ 0.063**

### Qué decir:
> "Los datos son muy raros: no sabemos qué significa ninguna columna — todo está anonimizado. Hay 86 pistas numéricas y un número de tiempo. Lo único que sabemos es el orden: lo que pasó antes viene primero. Hay cuatro 'horizontes': predecir a 1 paso, 3, 10 y 25 pasos adelante. Y la señal útil es **muy débil** — es como intentar escuchar un susurro en un estadio lleno de gente."

---

## Diapositiva 4 — Primer intento: Redes Neuronales (No funcionó)
**⏱ ~1.5 minutos**

### Qué poner:
- Dos columnas lado a lado con iconos de cerebro (MLP) y cadena (LSTM):

| | MLP (Red Densa) | LSTM (Red Recurrente) |
|---|---|---|
| Parámetros | ~millones | 412,315 |
| Épocas entrenadas | 50 | 100 |
| **Score público** | **0.0634** | **0.0891** |
| vs. Meta | ❌ muy lejos | ❌ muy lejos |

- Imagen humorística: un robot tratando de armar un rompecabezas al revés
- **3 razones del fracaso** (bullets cortos):
  1. Entrenamos solo con el 25% de los datos
  2. Los valores extremos dominan el error
  3. No les dimos información del orden del tiempo

### Qué decir:
> "Primero intentamos con redes neuronales — las mismas que usan para reconocer caras o traducir idiomas. Probamos dos tipos: una red 'densa' (MLP) y una red con memoria (LSTM). Ambas fallaron. El MLP aprendió su mejor respuesta en el primer intento y nunca mejoró. Sacamos tres lecciones: si no le das suficientes datos, si hay valores muy extremos, y si no le explicas el orden del tiempo — las redes neuronales no pueden con series financieras tan ruidosas."

---

## Diapositiva 5 — La Solución: LightGBM
**⏱ ~1.5 minutos**

### Qué poner:
- **Analogía visual:** un árbol de decisiones dibujado como un árbol real con ramas ("¿feature_al > 0? → izquierda/derecha")
- Texto clave:
  - LightGBM = **muchos árboles de decisión que se corrigen entre sí**
  - Aprende de los errores del árbol anterior (Gradient Boosting)
  - No entiende el tiempo por sí solo → **nosotros le damos el pasado como features**
- Caja resaltada: "¿Por qué funciona bien aquí?"
  - ✅ Datos tabulares con mucho ruido → LightGBM gana (igual que en M5 de Kaggle)
  - ✅ Rápido y estable
  - ✅ Maneja features categóricas nativamente

### Qué decir:
> "Cambié la estrategia completamente. LightGBM es como tener mil adivinos en fila. El primero intenta, se equivoca un poco. El siguiente se enfoca en corregir ese error. Y así uno tras otro. Al final, combinan sus respuestas. No entienden el tiempo solos, así que nosotros construimos 'pistas del pasado' — como promedios recientes y valores de hace unos pasos — y se las damos como entrada. Esto es exactamente lo que ganó la competencia M5 de Walmart en Kaggle."

---

## Diapositiva 6 — Cómo evitamos hacer trampa (Data Leakage)
**⏱ ~1 minuto**

### Qué poner:
- **Regla visual clara** con timeline:

```
─────────────────────────────────────────────────────
  ts_index: 1 ──────────── 3500 | 3503 ──── 4376
             [  TRAIN         ]  [ VAL ]  [ TEST  ]
                                   ↑
                        Solo miramos hasta aquí
                        cuando calculamos features
─────────────────────────────────────────────────────
```

- Dos ejemplos concretos:
  - ✅ `promedio_de_ayer = media(datos hasta t-1)` → OK
  - ❌ `promedio_de_mañana = media(datos hasta t+1)` → TRAMPA
- Dato curioso: mover el umbral de 3500 → 3503 (solo 3 filas de diferencia) **subió el score de 0.3212 a 0.3377**

### Qué decir:
> "La regla más importante de la competencia: **nunca puedes mirar el futuro**. Si calculas el promedio de una serie usando datos de mañana para predecir hoy, estás haciendo trampa — igual que leer las respuestas del examen antes de tomarlo. Verificamos esto computacionalmente: cero fugas de datos. Descubrimos algo curioso: mover el punto de corte de validación solo 3 filas — de la fila 3500 a la 3503 — subió nuestro score en 0.016 puntos. Así de sensible es este problema."

---

## Diapositiva 7 — Nuestra Mejor Solución: Ensemble Dual
**⏱ ~2 minutos**

### Qué poner:
- Diagrama de flujo visual simple:

```
         Datos de entrada (1.4M filas de test)
                        │
          ┌─────────────┴─────────────┐
          │                           │
  POOL A: Por Horizonte         POOL B: Por Categoría
  4 modelos (h=1,3,10,25)       5 modelos (subcategorías)
  Score promedio: 0.195         Score promedio: 0.254
          │                           │
          └─────────────┬─────────────┘
                        │
              Mezcla ponderada (potencia 2.5)
              67% Pool B + 33% Pool A
                        │
                  Score: 0.3429 ✅
```

- Fórmula simplificada:
  - "El modelo que funcionó mejor en validación, recibe **más peso** en la predicción final"
  - Peso = (score_validación)^2.5

### Qué decir:
> "Nuestra mejor solución entrena **9 modelos en total**: 4 especializados en cada horizonte de tiempo, y 5 especializados en cada categoría de serie. Después, para cada predicción, mezclamos ambas opiniones — pero el que demostró ser mejor en validación tiene más voz. La potencia 2.5 hace que las diferencias se amplíen: un modelo con score 0.31 vs 0.07 termina aportando el 97% del peso. El resultado: score público de **0.3429**, superando nuestra meta de 0.30."

---

## Diapositiva 8 — Demo en vivo: el Notebook corriendo
**⏱ ~2 minutos**

### Qué poner:
- Título: **"Veámoslo en acción"**
- Captura de pantalla (o ventana en vivo) del notebook `score-0-3429.ipynb` mostrando:
  1. La carga de datos: `Train shape: (5337414, 94)`
  2. El entrenamiento de un modelo por horizonte con su score
  3. El blending final y las predicciones de ejemplo:
     ```
     Row 1:
       ID: 10BAVIDU__E9OOLYU3__PZ9S1Z4V__1__3602
       Horizon 1 (val=0.073, weight=0.026) → Pred: 0.084
       PZ9S1Z4V  (val=0.313, weight=0.974) → Pred: 1.746
       ✅ FINAL: 1.703
     ```
  4. Las estadísticas finales de la submission

### Qué decir:
> "Aquí está el notebook real que enviamos a Kaggle. [Mostrar pantalla] Primero carga los 5.3 millones de filas de datos. Luego entrena los 9 modelos secuencialmente — cada uno con su propio conjunto de entrenamiento y validación temporal. Finalmente mezcla las predicciones con los pesos calculados. Fíjense en este ejemplo: para la fila con horizonte 1, el modelo por categoría tiene un peso del 97% porque fue mucho más preciso. El resultado son 1.4 millones de predicciones listas para subir."

---

## Diapositiva 9 — Resultados y Análisis
**⏱ ~2 minutos**

### Qué poner:
- Tabla comparativa de todos los experimentos (columna de barras de progreso visuales):

| # | Modelo | Score Público |
|---|---|---|
| MLP | Red neuronal densa | 0.063 ░░░░░░░░░░ |
| LSTM | Red recurrente | 0.089 ░░░░░░░░░░ |
| Exp 1 | LightGBM + 183 features | 0.268 ██████░░░░ |
| Exp 3 | + Polars + filtro outliers | 0.271 ██████░░░░ |
| Exp 4 | + Stacking XGB + Ridge | 0.282 ███████░░░ |
| Exp 5 | Minimalista + IDs categoría | 0.321 ████████░░ |
| Exp 6 | + Umbral validación ajustado | 0.338 ████████░░ |
| **Exp 7** | **Ensemble dual (ganador)** | **0.343 █████████░** |

- **3 hallazgos clave** destacados:
  1. LightGBM superó a redes neuronales por **4x**
  2. Agregar la **identidad de la serie** como feature fue el mayor salto (+0.04)
  3. Más complejidad ≠ mejor resultado (el modelo minimalista ganó al stacking)

### Qué decir:
> "Tres lecciones que nos llevamos. Primero: en datos financieros ruidosos y tabulares, LightGBM le gana a las redes neuronales por lejos — cuatro veces mejor en nuestro caso. Segundo: el descubrimiento más valioso fue pasar los identificadores de cada serie (código, subcódigo, categoría) como variables categóricas. LightGBM aprendió un 'efecto fijo' por entidad automáticamente — eso fue el mayor salto de todo el proyecto. Tercero: más complejidad no siempre ayuda. El modelo con stacking de XGBoost y Ridge fue superado por uno más simple que simplemente conocía la identidad de cada serie."

---

## Diapositiva 10 — Conclusiones y Próximos Pasos
**⏱ ~1 minuto**

### Qué poner:
- **Lo que logramos** (en verde):
  - ✅ Score 0.3429 — superamos la meta de 0.30
  - ✅ Zero data leakage verificado
  - ✅ Pipeline reproducible bit-a-bit (semilla fija, determinístico)

- **Lo que aprendimos** (bullets cortos):
  - "La identidad importa más que las features temporales complejas"
  - "El horizonte corto (h=1) es el más difícil — demasiado ruido"
  - "Validar bien es más importante que un modelo más complejo"

- **¿Qué haríamos si tuviéramos más tiempo?**
  - Transformers especializados en series financieras (CATS, TimeXer)
  - Optimización bayesiana de hiperparámetros por horizonte
  - Ensembles con más modelos por sub-categoría

- Frase de cierre en grande:
  > *"No hay almuerzo gratis — pero sí hay estrategias inteligentes."*

### Qué decir:
> "Logramos nuestro objetivo: score de 0.3429 en el leaderboard público de Kaggle. El mensaje más importante del proyecto es este: en problemas de datos reales, a veces la clave no es el modelo más sofisticado, sino entender qué información le das. Agregar la 'identidad' de cada serie fue más valioso que cientos de features elaboradas. Y validar correctamente, sin trampa, es lo que separa una solución robusta de una que solo funciona en los datos que ya conoces. Gracias."

---

## Cronograma de la presentación

| Diapositiva | Tema | Tiempo |
|---|---|---|
| 1 | Portada | 0:30 |
| 2 | El Problema | 2:00 |
| 3 | Los Datos | 1:30 |
| 4 | Redes Neuronales (fracaso) | 1:30 |
| 5 | LightGBM | 1:30 |
| 6 | Data Leakage | 1:00 |
| 7 | Ensemble Dual (solución) | 2:00 |
| 8 | Demo en vivo | 2:00 |
| 9 | Resultados y Análisis | 2:00 |
| 10 | Conclusiones | 1:00 |
| **Total** | | **~15:00** |

---

## Tips para el día de la presentación

- **Demo (Diap. 8):** Abre el notebook `experimentos/score-0-3429.ipynb` antes de la presentación con los outputs ya ejecutados. No corras el modelo en vivo — tarda ~6 minutos. Solo muestra el output ya generado.
- **Pregunta trampa frecuente:** "¿Por qué no usaron más datos?" → Respuesta: sí usamos el 100% en LightGBM (las redes neuronales usaron 25% por limitaciones de memoria).
- **Pregunta sobre el score:** El score 0.3429 está calculado sobre el leaderboard público, que es un subconjunto del test real — el score final puede variar.
- **Analogía del niño de 5 años para cada concepto:**
  - Series de tiempo → "¿Cuántos dulces habrá mañana en el frasco?"
  - Data leakage → "Leer las respuestas antes del examen"
  - Ensemble → "Preguntar a 9 amigos y promediar su opinión"
  - Horizonte → "¿Adivinas qué pasa mañana, en 3 días, en 10 días, en 25 días?"
  - Score → "Qué tan lejos estás de la respuesta correcta (0 = pésimo, 1 = perfecto)"
