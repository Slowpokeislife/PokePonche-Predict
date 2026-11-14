"""PokePonche Predict: evaluación temporal CatBoost / Gradient Boosting / PyTorch MLP.
Ejecutar desde raíz del proyecto: py scripts/train_etapa2.py --mode ambos
"""
from __future__ import annotations
import argparse
import json
import random
import re
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from fases_eventos import add_phases, get_family

SEED = 42
MAJOR_PATTERN = re.compile(r"(?:go\s*fest|go\s*tour|wild\s*area|expedici[oó]n\s*silvestre|[aá]rea\s*silvestre)", re.I)
EXCLUDE_COLUMN_PATTERNS = ('checkins', 'rsvp', 'username', 'user_id', 'evidence', 'evidencia', 'url')
BASE_NUMERICS = [
    'is_global_major_event', 'is_global_pre_event', 'days_until_main_event', 'is_weekend', 'year_offset', 'month', 'day_of_week',
    'week_of_year', 'is_holiday_season', 'event_sequence', 'days_since_first_event',
    'prior_events_same_type', 'hist_last_5_mean', 'hist_last_10_mean',
    'hist_last_10_median', 'hist_same_type_last_5_mean',
    'hist_same_type_last_5_median', 'days_since_previous_event',
    'days_since_previous_same_type', 'prev_event_checkins',
    'has_paid_bonus_evidence', 'has_free_bonus_evidence',
    'has_assumed_free_bonus', 'has_exclusive_move_section',
    'has_shiny_boost_evidence', 'has_pokemon_debut_evidence',
    'has_paid_research_evidence', 'pokemon_label_missing',
    'primary_pokemon_count',
]
# Post-event/exploratory variables, NEVER used in prospective model.
RETRO_NUMERICS = [
    'weather_temp_mean_c', 'weather_temp_max_c', 'weather_temp_min_c',
    'weather_humidity_mean_pct', 'weather_precipitation_mm_day',
    'weather_rain_mm_day', 'weather_rain_hours', 'weather_cloud_cover_mean_pct',
    'weather_wind_mean_kmh',
    'popularity_pct_mean_current_exploratory',
    'pve_best_rank_current_exploratory', 'pve_best_edps_current_exploratory',
    'pvp_great_best_rank_current_exploratory',
    'pvp_ultra_best_rank_current_exploratory',
    'pvp_master_best_rank_current_exploratory',
]
CAT_COLS = ['tipo_campfire', 'major_event_subtype', 'event_phase', 'month_label']

def major_kind(name: str, typ: str) -> str:
    """Compatibilidad con importaciones del clasificador anterior."""
    return get_family(name,typ)


def build_features(raw: pd.DataFrame) -> pd.DataFrame:
    """History is calculated using strictly earlier DAYS (not same-day targets)."""
    df = raw.copy()
    required = ['event_id','fecha_puebla','tipo_campfire','nombre_campfire','checkins']
    missing = [c for c in required if c not in df]
    if missing: raise ValueError(f'Faltan columnas en el maestro: {missing}')
    df['date'] = pd.to_datetime(df['fecha_puebla'], format='%d/%m/%Y', errors='raise')
    if df['event_id'].isna().any() or df['event_id'].duplicated().any():
        raise ValueError('event_id nulo o repetido: maestro debe tener una fila por quedada')
    df['checkins'] = pd.to_numeric(df['checkins'], errors='raise')
    if df['checkins'].isna().any() or (df['checkins'] < 0).any():
        raise ValueError('checkins faltantes o negativos')
    df = df.sort_values(['date','event_id'], kind='stable').reset_index(drop=True)
    df['tipo_campfire'] = df['tipo_campfire'].fillna('sin_tipo').astype(str)
    df = add_phases(df)
    df['is_weekend'] = (df['date'].dt.dayofweek >= 5).astype(int)
    df['month'] = df['date'].dt.month
    df['month_label'] = df['month'].map(lambda x:f'mes_{x:02}')
    df['year_offset'] = df['date'].dt.year - df['date'].dt.year.min()
    df['day_of_week'] = df['date'].dt.dayofweek
    df['week_of_year'] = df['date'].dt.isocalendar().week.astype(int)
    df['is_holiday_season'] = df['month'].isin([11,12]).astype(int)
    df['days_since_first_event'] = (df['date'] - df['date'].min()).dt.days

    # Only strictly older date groups may be used in historical aggregates.
    history = []
    previous = []
    prev_date = None
    df['event_sequence'] = np.arange(len(df))
    for date, idx in df.groupby('date', sort=True).groups.items():
        for i in idx:
            row = df.loc[i]
            typ = row['tipo_campfire']
            same = [h for h in history if h[1] == typ]
            vals = [h[2] for h in history]
            svals = [h[2] for h in same]
            prev_same_date = same[-1][0] if same else None
            previous.append((i, {
                'prior_events_same_type':len(same),
                'hist_last_5_mean':float(np.mean(vals[-5:])) if vals else np.nan,
                'hist_last_10_mean':float(np.mean(vals[-10:])) if vals else np.nan,
                'hist_last_10_median':float(np.median(vals[-10:])) if vals else np.nan,
                'hist_same_type_last_5_mean':float(np.mean(svals[-5:])) if svals else np.nan,
                'hist_same_type_last_5_median':float(np.median(svals[-5:])) if svals else np.nan,
                'days_since_previous_event':(date - prev_date).days if prev_date is not None else np.nan,
                'days_since_previous_same_type':(date-prev_same_date).days if prev_same_date is not None else np.nan,
                'prev_event_checkins':vals[-1] if vals else np.nan,
            }))
        for i in idx:
            history.append((date, df.at[i,'tipo_campfire'],df.at[i,'checkins']))
        prev_date = date
    for i, features in previous:
        for key, value in features.items(): df.at[i,key] = value
    return df


def chronological_split(df):
    days = np.sort(df['date'].unique())
    if len(days) < 20: raise ValueError('Se necesitan al menos 20 fechas diferentes')
    i, j = int(len(days)*0.70), int(len(days)*0.85)
    if i <= 0 or j <= i or j >= len(days): raise ValueError('Partición temporal inválida')
    train = df['date'] < days[i]
    val = (df['date'] >= days[i]) & (df['date'] < days[j])
    test = df['date'] >= days[j]
    if not (train.any() and val.any() and test.any()): raise ValueError('Alguna partición quedó vacía')
    return train.to_numpy(),val.to_numpy(),test.to_numpy()


def select_features(df,mode):
    numeric = [x for x in BASE_NUMERICS if x in df]
    if mode=='retrospectivo': numeric += [x for x in RETRO_NUMERICS if x in df]
    # Explicit allowlist avoids leakage from after-event statuses, evidence, targets and IDs.
    cat = [x for x in CAT_COLS if x in df]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df[cat] = df[cat].fillna('desconocido').astype(str)
    return numeric,cat


def sklearn_pipeline(numeric,cat):
    transform = ColumnTransformer([
        ('num', Pipeline([('fill',SimpleImputer(strategy='median', add_indicator=True)),('scale',StandardScaler())]), numeric),
        ('cat',OneHotEncoder(handle_unknown='ignore',sparse_output=False),cat),
    ],sparse_threshold=0.0)
    return transform


def evaluate(y, pred):
    y=np.asarray(y,dtype=float);pred=np.clip(np.asarray(pred,dtype=float),0,None)
    return {'MAE':float(mean_absolute_error(y,pred)), 'RMSE':float(np.sqrt(mean_squared_error(y,pred))),
            'R2':float(r2_score(y,pred)), 'bias_pred_minus_real':float(np.mean(pred-y))}


def train_catboost(df,numeric,cat,tr,va,te):
    try: from catboost import CatBoostRegressor
    except ImportError as e: raise RuntimeError('Instala catboost con pip install -r requirements_etapa2.txt') from e
    cols=numeric+cat
    x=df[cols].copy()
    for c in numeric:x[c]=x[c].fillna(-99999).astype(float)
    for c in cat:x[c]=x[c].fillna('desconocido').astype(str)
    model=CatBoostRegressor(iterations=1000,learning_rate=0.04,depth=4,l2_leaf_reg=8,
        loss_function='MAE',random_seed=SEED,verbose=False,thread_count=2)
    model.fit(x.loc[tr],df.loc[tr,'checkins'],cat_features=cat,
              eval_set=(x.loc[va],df.loc[va,'checkins']),early_stopping_rounds=80)
    preds=np.clip(model.predict(x.loc[te]),0,None)
    return model,preds,cols


def train_gb(df,numeric,cat,tr,va,te):
    cols=numeric+cat
    transformer=sklearn_pipeline(numeric,cat)
    xtrain=transformer.fit_transform(df.loc[tr,cols])
    xval=transformer.transform(df.loc[va,cols])
    xtest=transformer.transform(df.loc[te,cols])
    # Use validation (not test) to select n_iter.
    best=None
    for n in [80,160,300]:
        m=HistGradientBoostingRegressor(max_iter=n,max_leaf_nodes=10,learning_rate=0.05,
             l2_regularization=10,min_samples_leaf=12,random_state=SEED,early_stopping=False)
        m.fit(xtrain,df.loc[tr,'checkins'])
        val_loss=mean_absolute_error(df.loc[va,'checkins'],np.clip(m.predict(xval),0,None))
        if best is None or val_loss<best[0]:best=(val_loss,m,n)
    _,model,n=best
    return {'preprocess':transformer,'model':model,'n_iter':n},np.clip(model.predict(xtest),0,None),cols


def train_mlp(df,numeric,cat,tr,va,te,epochs=350):
    try:
        import torch
        from torch import nn
    except ImportError as e: raise RuntimeError('PyTorch no instalado. Ver README.') from e
    torch.manual_seed(SEED)
    torch.set_num_threads(2)
    cols=numeric+cat
    transformer=sklearn_pipeline(numeric,cat)
    xtr=transformer.fit_transform(df.loc[tr,cols]).astype('float32')
    xva=transformer.transform(df.loc[va,cols]).astype('float32')
    xte=transformer.transform(df.loc[te,cols]).astype('float32')
    ytr=np.log1p(df.loc[tr,'checkins'].to_numpy(dtype='float32'))
    yva=df.loc[va,'checkins'].to_numpy(dtype='float32')
    xt=torch.tensor(xtr);yt=torch.tensor(ytr).reshape(-1,1)
    xv=torch.tensor(xva)
    network=nn.Sequential(nn.Linear(xtr.shape[1],32),nn.ReLU(),nn.Dropout(0.10),
                          nn.Linear(32,16),nn.ReLU(),nn.Linear(16,1))
    opt=torch.optim.AdamW(network.parameters(),lr=0.001,weight_decay=0.02)
    loss_fn=nn.SmoothL1Loss()
    best=float('inf');state=None;wait=0;best_epoch=0
    for epoch in range(epochs):
        network.train()
        order=torch.randperm(len(xt))
        for ids in order.split(32):
            opt.zero_grad();loss=loss_fn(network(xt[ids]),yt[ids]);loss.backward();opt.step()
        network.eval()
        with torch.no_grad():
            val_pred=np.clip(np.expm1(network(xv).numpy().ravel()),0,None)
        score=mean_absolute_error(yva,val_pred)
        if score < best-0.01:
            best=score;state={k:v.detach().clone() for k,v in network.state_dict().items()};wait=0;best_epoch=epoch+1
        else:
            wait+=1
            if wait>=45:break
    network.load_state_dict(state);network.eval()
    with torch.no_grad():pred=np.clip(np.expm1(network(torch.tensor(xte)).numpy().ravel()),0,None)
    return {'network':network,'transformer':transformer,'columns':cols,'best_epoch':best_epoch},pred,cols


def save_mlp(obj,out):
    import torch
    torch.save({'state_dict':obj['network'].state_dict(), 'input_dim':obj['network'][0].in_features,
                'best_epoch':obj['best_epoch']},out/'mlp_weights.pt')
    joblib.dump({'transformer':obj['transformer'],'columns':obj['columns']},out/'mlp_preprocesador.joblib')


def run_one(raw,mode,out_root,epochs):
    df=build_features(raw)
    numeric,cat=select_features(df,mode)
    tr,va,te=chronological_split(df)
    out=out_root/mode;out.mkdir(parents=True,exist_ok=True)
    splits=np.select([tr,va,te],['train','validation','test'],default='ERROR')
    pd.DataFrame({'event_id':df['event_id'],'fecha_puebla':df['fecha_puebla'],'split':splits}).to_csv(out/'particiones.csv',index=False,encoding='utf-8-sig')
    y=df.loc[te,'checkins'].to_numpy()
    dates=df.loc[te,'date']
    metrics=[];results=df.loc[te,['event_id','fecha_puebla','tipo_campfire','nombre_campfire','major_event_subtype','event_phase','is_global_pre_event','is_global_major_event','checkins']].copy()
    results=results.rename(columns={'checkins':'real'})
    baseline=float(df.loc[tr,'checkins'].median())
    results['pred_baseline']=baseline
    metrics.append({'modelo':'baseline_mediana_train',**evaluate(y,np.full(len(y),baseline))})

    model,pred,cols=train_gb(df,numeric,cat,tr,va,te)
    results['pred_gradient_boosting']=pred
    metrics.append({'modelo':'gradient_boosting',**evaluate(y,pred)})
    joblib.dump(model,out/'gradient_boosting.joblib')

    model,pred,cols=train_catboost(df,numeric,cat,tr,va,te)
    results['pred_catboost']=pred
    metrics.append({'modelo':'catboost',**evaluate(y,pred)})
    model.save_model(str(out/'catboost.cbm'))
    pd.DataFrame({'feature':cols,'importance':model.get_feature_importance()}).sort_values('importance',ascending=False).to_csv(out/'catboost_importancias.csv',index=False,encoding='utf-8-sig')

    model,pred,cols=train_mlp(df,numeric,cat,tr,va,te,epochs=epochs)
    results['pred_mlp']=pred
    metrics.append({'modelo':'mlp_pytorch',**evaluate(y,pred)})
    save_mlp(model,out)

    mdf=pd.DataFrame(metrics).sort_values('MAE');mdf.to_csv(out/'metricas.csv',index=False,encoding='utf-8-sig')
    results.to_csv(out/'predicciones_test.csv',index=False,encoding='utf-8-sig')
    subgroup=[]
    for kind,sub in [('global_main',results[results.event_phase=='main_event']),('global_pre',results[results.event_phase=='pre_event']),('otros',results[results.event_phase=='not_global'])]:
        if len(sub)==0:continue
        for modelname in ['baseline','gradient_boosting','catboost','mlp']:
            error=evaluate(sub.real,sub[f'pred_{modelname}'])
            subgroup.append({'grupo':kind,'n':len(sub),'modelo':modelname,**error})
    for typ,sub in results.groupby('tipo_campfire'):
        for modelname in ['baseline','gradient_boosting','catboost','mlp']:
            error=evaluate(sub.real,sub[f'pred_{modelname}']) if len(sub)>1 else {'MAE':float(abs(sub[f'pred_{modelname}'].iloc[0]-sub.real.iloc[0])),'RMSE':np.nan,'R2':np.nan,'bias_pred_minus_real':float(sub[f'pred_{modelname}'].iloc[0]-sub.real.iloc[0])}
            subgroup.append({'grupo':f'tipo:{typ}','n':len(sub),'modelo':modelname,**error})
    pd.DataFrame(subgroup).to_csv(out/'metricas_por_tipo.csv',index=False,encoding='utf-8-sig')
    report={
        'modo':mode,'total':len(df),'train':int(tr.sum()),'validation':int(va.sum()),'test':int(te.sum()),
        'periods':{s:{'first':df.loc[mask,'date'].min().strftime('%Y-%m-%d'),'last':df.loc[mask,'date'].max().strftime('%Y-%m-%d')} for s,mask in [('train',tr),('validation',va),('test',te)]},
        'n_major_test':int(results.is_global_major_event.sum()),'n_major_total':int(df.is_global_major_event.sum()),
        'n_pre_test':int(results.is_global_pre_event.sum()),'n_pre_total':int(df.is_global_pre_event.sum()),
        'features_numeric':numeric,'features_categorical':cat,
        'best_model_by_test_mae_for_reporting_only':str(mdf.iloc[0]['modelo']),
        'IMPORTANT':'No selecciones hiperparámetros según TEST. PvP/PvE 2026 solo en retrospectivo exploratorio; clima observado no es pronóstico. Historias de asistencia usan fechas anteriores exclusivamente.'
    }
    (out/'reporte.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print(f'\n=== {mode.upper()} ===\n',mdf.to_string(index=False,float_format=lambda x:f'{x:.2f}'))
    print(f'Particiones {tr.sum()} / {va.sum()} / {te.sum()}, globales principales en prueba: {results.is_global_major_event.sum()}, preparatorios: {results.is_global_pre_event.sum()}')


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--csv',default='dataset_maestro_548.csv')
    p.add_argument('--mode',choices=['anticipado','retrospectivo','ambos'],default='ambos')
    p.add_argument('--out',default='resultados_etapa2')
    p.add_argument('--epochs',type=int,default=350)
    args=p.parse_args()
    np.random.seed(SEED);random.seed(SEED)
    data=pd.read_csv(args.csv,low_memory=False)
    modes=['anticipado','retrospectivo'] if args.mode=='ambos' else [args.mode]
    for mode in modes: run_one(data,mode,Path(args.out),args.epochs)
    print('\nFinalizado. Revisa resultados_etapa2/<modo>/metricas.csv y predicciones_test.csv')

if __name__=='__main__': main()
