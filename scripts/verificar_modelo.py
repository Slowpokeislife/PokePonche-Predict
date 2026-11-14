"""Carga el modelo guardado, regenera features públicas y verifica las predicciones en test.
Ejecutar desde la raíz: python scripts/verificar_modelo.py
"""
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from train_etapa5 import prepare, make_matrix, chronological_split
from train_etapa5_rsvp import add_rsvp, prior_ratios
ROOT=Path(__file__).resolve().parents[1]
master=ROOT/'data/dataset_maestro_548.csv'
people=ROOT/'data/registros_usuarios_pseudonimizados.csv'
data=add_rsvp(prepare(master,people),people)
x,cats=make_matrix(data,'anticipado')
x['interesados_total_proxy']=data.interesados_total_proxy.to_numpy(float)
x['conversion_historica_anterior']=prior_ratios(data)
_,_,te=chronological_split(data)
model=CatBoostRegressor()
model.load_model(str(ROOT/'models/modelo_catboost.cbm'))
# winner 'conversion' has special postprocessing learned during training
p=np.asarray(model.predict(x.loc[te].drop(columns=['interesados_total_proxy'])),float)
p=np.clip(p,0,1)*x.loc[te,'interesados_total_proxy'].to_numpy(float)
p=np.maximum(p,0)
mae=np.abs(p-data.loc[te,'checkins'].to_numpy(float)).mean()
print(f'MAE regenerado: {mae:.4f}, n={te.sum()}')
print('Comparar con el histórico publicado: ~6.75 personas')
