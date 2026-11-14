r"""Reporte visual de PokePonche Predict, Etapa 5.

Colocar este archivo en scripts/reporte_etapa5.py y ejecutarlo desde la raiz
con: .\.venv\Scripts\python.exe scripts\reporte_etapa5.py

Lee resultados_etapa5_rsvp/predicciones_test.csv o, si la carpeta no
existe, resultados_etapa5_rsvp.zip situado en la raiz.
"""
from pathlib import Path
import io
import json
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path.cwd()
OUT = ROOT / 'assets'
OUT.mkdir(exist_ok=True)
PRED = ROOT / 'results' / 'predicciones_test.csv'
ZIP = ROOT / 'resultados_etapa5_rsvp.zip'

if PRED.exists():
    df = pd.read_csv(PRED)
elif ZIP.exists():
    with zipfile.ZipFile(ZIP) as z:
        matches = [x for x in z.namelist() if x.endswith('/predicciones_test.csv') or x == 'predicciones_test.csv']
        if not matches:
            raise FileNotFoundError('No hay predicciones_test.csv en el ZIP')
        df = pd.read_csv(io.BytesIO(z.read(matches[0])))
else:
    raise FileNotFoundError('Falta resultados_etapa5_rsvp/predicciones_test.csv o resultados_etapa5_rsvp.zip')

required = {'fecha_puebla', 'checkins', 'prediccion', 'nombre_campfire'}
missing = required - set(df.columns)
if missing:
    raise ValueError(f'Columnas faltantes: {sorted(missing)}')

df['fecha'] = pd.to_datetime(df['fecha_puebla'], dayfirst=True, errors='raise')
df['checkins'] = pd.to_numeric(df['checkins'], errors='raise')
df['prediccion'] = pd.to_numeric(df['prediccion'], errors='raise')
df = df.sort_values(['fecha','nombre_campfire']).reset_index(drop=True)
df['error'] = df['prediccion'] - df['checkins']
df['error_abs'] = df['error'].abs()
err = df['error_abs']
actual = df['checkins'].to_numpy()
pred = df['prediccion'].to_numpy()
n = len(df)
mae = float(err.mean())
rmse = float(np.sqrt(np.mean((pred - actual)**2)))
wape = float(100 * err.sum() / actual.sum()) if actual.sum() else float('nan')
bias = float(df['error'].mean())
r2 = float(1 - np.sum((pred - actual)**2) / np.sum((actual - actual.mean())**2)) if n > 1 and np.var(actual) else float('nan')
qs = {k: float(err.quantile(p)) for k,p in [('q1',.25),('mediana',.5),('q3',.75),('p90',.90),('p95',.95)]}

# 1. Tendencia por quedada (no se agregan fechas: puede haber varias por dia)
fig, ax = plt.subplots(figsize=(15,5.5))
x = np.arange(n)
ax.plot(x, actual, '-o', markersize=3.5, linewidth=1.4, label='Check-ins reales')
ax.plot(x, pred, '-o', markersize=3.0, linewidth=1.4, label='Predicción Etapa 5')
positions = np.unique(np.linspace(0,n-1,min(n,10)).astype(int))
ax.set_xticks(positions, [df.iloc[i]['fecha'].strftime('%d/%m/%Y') for i in positions], rotation=30, ha='right')
ax.set_xlabel('Quedadas de prueba ordenadas cronológicamente')
ax.set_ylabel('Número de check-ins')
ax.set_title('Asistencia real vs. predicha · Etapa 5')
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'01_tendencia_real_vs_prediccion.png', dpi=220)
plt.close(fig)

# 2. Scatter + diagonal ideal
fig, ax = plt.subplots(figsize=(7,7))
lim = float(max(actual.max(), pred.max()) * 1.05)
ax.scatter(actual, pred, alpha=.72, s=40)
ax.plot([0,lim], [0,lim], '--', linewidth=1.2, label='Predicción perfecta')
ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='Check-ins reales',ylabel='Predicción')
ax.set_title(f'Real vs. predicho · MAE {mae:.2f} · R² {r2:.3f}')
ax.legend()
ax.grid(alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'02_dispersion_real_vs_prediccion.png', dpi=220)
plt.close(fig)

# 4. Boxplot
fig,ax=plt.subplots(figsize=(10,3.4))
ax.boxplot(err,vert=False,showmeans=True,patch_artist=True,
           boxprops={'facecolor':'#b5c9df','alpha':.6},
           medianprops={'color':'#8a3f12','linewidth':2})
ax.set(xlabel='Error absoluto (personas)',yticks=[],title='Cuartiles del error absoluto · Etapa 5')
ax.grid(axis='x',alpha=.2)
fig.tight_layout()
fig.savefig(OUT/'04_boxplot_error.png',dpi=220)
plt.close(fig)

metrics={'evaluadas':n,'mae':mae,'rmse':rmse,'wape_porcentaje':wape,'r2':r2,'sesgo_medio':bias,**qs,
         'suma_checkins_reales':int(actual.sum()),'aciertos_error_10_o_menos':int((err<=10).sum()),
         'observacion':'Interesados reconstruidos del CSV final como CHECKED_IN + ACCEPTED; no son necesariamente capturas al inicio.'}
(OUT/'metricas_resumen.json').write_text(json.dumps(metrics,ensure_ascii=False,indent=2),encoding='utf-8')

print('Gráficos creados en',OUT)
print(json.dumps(metrics,ensure_ascii=False,indent=2))
