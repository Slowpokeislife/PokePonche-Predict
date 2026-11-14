"""Etapa 3: validación temporal, Optuna, ponderación y ajustes por familia global.
La partición final de test no participa en selección de hiperparámetros.
El historial de asistencia está disponible solo de días anteriores (evaluación secuencial).
"""
from __future__ import annotations
import argparse
import json
import sys
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import optuna

from train_etapa2 import build_features, chronological_split, select_features, BASE_NUMERICS, RETRO_NUMERICS

SEED = 42

def edition_keys(df):
    """Agrupa días de la misma edición global; no mezcla fases ni familias.
    Días principales consecutivos (separados por <=7 días) en el mismo año => misma edición.
    """
    keys=[]; last={}; counts={}
    for r in df.itertuples(index=False):
        dt=pd.Timestamp(r.date)
        phase=str(r.event_phase)
        fam=str(r.event_family)
        if phase != 'main_event' or fam=='no_global':
            keys.append('regular:'+str(r.event_id));continue
        typ=(fam,dt.year)
        if typ not in last or (dt-last[typ]).days>7:
            counts[typ]=counts.get(typ,0)+1
        last[typ]=dt
        keys.append(f'global:{fam}:{dt.year}:{counts[typ]}')
    return np.asarray(keys)


def dev_cv_folds(df,dev_mask,n_folds=3):
    """Validación con ventanas crecientes, grupos de edición global no divididos."""
    dev=df.loc[dev_mask].copy()
    daygroup=dev.groupby('date',sort=True).size()
    dates=np.array(daygroup.index)
    if len(dates)<80:raise ValueError('Fechas insuficientes para validación temporal')
    keys=edition_keys(df)
    all_dates=df['date'].to_numpy()
    folds=[]
    # 3 ventanas aproximadamente al 55-67, 67-79, 79-100% del periodo de desarrollo.
    borders=np.linspace(.55,1.0,n_folds+1)
    for i in range(n_folds):
        start=dates[min(int(len(dates)*borders[i]),len(dates)-2)]
        end=dates[min(int(len(dates)*borders[i+1]),len(dates)-1)] if i<n_folds-1 else dates[-1]+pd.Timedelta(days=1)
        training=dev_mask & (all_dates<start)
        valid=dev_mask & (all_dates>=start) & (all_dates<end)
        # Si alguna edición empieza en train y continúa en valid, moverla por completo a valid;
        # si abarca valid y fuera, quitarla de la validación (protege edición).
        contaminated=set(keys[training]).intersection(set(keys[valid]))
        if contaminated:
            training &= ~np.isin(keys,list(contaminated))
        if training.sum()<50 or valid.sum()<15:continue
        assert max(all_dates[training]) < min(all_dates[valid])
        assert not (set(keys[training]) & set(keys[valid]))
        folds.append((training,valid))
    if len(folds)<2:raise ValueError(f'Solo {len(folds)} folds válidos; se necesitan >=2')
    return folds


def features_for_mode(df,mode):
    numeric,cat=select_features(df,mode)
    x=df[numeric+cat].copy()
    for c in numeric:x[c]=x[c].fillna(-99999).astype(float)
    for c in cat:x[c]=x[c].fillna('desconocido').astype(str)
    return x,numeric,cat


def model_params(params=None):
    defaults=dict(iterations=650,depth=4,learning_rate=.04,l2_leaf_reg=8,random_strength=1.0,
                  loss_function='MAE',random_seed=SEED,verbose=False,thread_count=2,
                  allow_writing_files=False)
    if params:defaults.update(params)
    return defaults


def fit_predict(x,y,train,valid,cat,params,weight_mode='none',log_target=False):
    m=CatBoostRegressor(**model_params(params))
    yt=y[train]
    w=None
    if weight_mode=='high_attendance':
        # Límite definido SOLO usando train. Ponderación suave para no sobreajustar excepciones.
        threshold=np.quantile(yt,.80)
        w=np.where(yt>=threshold,2.0,1.0)
    target=np.log1p(yt) if log_target else yt
    m.fit(x.loc[train],target,cat_features=cat,sample_weight=w)
    p=m.predict(x.loc[valid]);p=np.expm1(p) if log_target else p
    return m,np.clip(np.asarray(p,float),0,None)


def shrinkage_correction(y_train,train_predictions,families,alpha=12.0,limit=100.0):
    """Corrección suavizada por familia, sin usar la verdad de validación/test."""
    residual=np.asarray(y_train,float)-np.asarray(train_predictions,float)
    out={}
    for f in ('go_fest','go_tour','wild_area'):
        selected=np.asarray(families)==f
        n=int(selected.sum())
        delta=float(np.clip(np.sum(residual[selected])/(n+alpha),-limit,limit)) if n else 0.0
        out[f]={'n':n,'offset':delta}
    return out


def specialized_predictions(model,x,y,train,valid,df,raw_pred,alpha=12.0):
    # Atención: residuos del entrenamiento in-sample están sesgados y solo sirven como
    # ajuste muy regularizado exploratorio. Elegir estrategia únicamente en CV.
    train_p=np.clip(model.predict(x.loc[train]),0,None)
    offsets=shrinkage_correction(y[train],train_p,df.loc[train,'event_family'],alpha=alpha)
    new=np.array(raw_pred,float)
    for i, f in enumerate(df.loc[valid,'event_family']):
        if f in offsets and df.loc[valid,'event_phase'].iloc[i]=='main_event':
            new[i]+=offsets[f]['offset']
    return np.clip(new,0,None),offsets


def evaluate(y,p):
    y=np.asarray(y,float);p=np.clip(np.asarray(p,float),0,None)
    return {'MAE':round(float(mean_absolute_error(y,p)),4),
            'RMSE':round(float(np.sqrt(mean_squared_error(y,p))),4),
            'R2':round(float(r2_score(y,p)),4) if len(y)>1 else None,
            'bias':round(float(np.mean(p-y)),4)}


def tune(x,df,cat,folds,trials,seed):
    y=df.checkins.to_numpy(float)
    def objective(trial):
        params={'iterations':trial.suggest_int('iterations',280,850,step=95),
                'depth':trial.suggest_int('depth',3,6),
                'learning_rate':trial.suggest_float('learning_rate',.015,.12,log=True),
                'l2_leaf_reg':trial.suggest_float('l2_leaf_reg',2,25,log=True),
                'random_strength':trial.suggest_float('random_strength',.1,3.0)}
        scores=[]
        for tr,va in folds:
            _,p=fit_predict(x,y,tr,va,cat,params)
            scores.append(mean_absolute_error(y[va],p))
        return float(np.mean(scores))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective,n_trials=trials,show_progress_bar=False)
    return study


def cv_strategy(x,df,cat,folds,params,strategy):
    y=df.checkins.to_numpy(float)
    stats=[]
    for k,(tr,va) in enumerate(folds,1):
        if strategy=='baseline_median':
            p=np.full(int(va.sum()),np.median(y[tr]))
        elif strategy=='baseline_same_type':
            # Histórico calculado con días anteriores, no actual ni futuros.
            p=df.loc[va,'hist_same_type_last_5_median'].to_numpy(float)
            missing=~np.isfinite(p)
            p[missing]=np.median(y[tr])
        else:
            weighted=strategy=='weighted'
            log_target=strategy=='log_target'
            model,p=fit_predict(x,y,tr,va,cat,params,
                                weight_mode='high_attendance' if weighted else 'none',log_target=log_target)
            if strategy=='global_adjusted':
                p,_=specialized_predictions(model,x,y,tr,va,df,p)
        score=evaluate(y[va],p)
        global_mask=(df.loc[va,'event_phase']=='main_event').to_numpy()
        score.update({'fold':k,'strategy':strategy,'n':int(va.sum()),
                      'global_main_n':int(global_mask.sum()),
                      'global_main_mae':round(float(mean_absolute_error(y[va][global_mask],p[global_mask])),4) if global_mask.any() else None})
        stats.append(score)
    return stats


def run_mode(df,mode,out,trials):
    out.mkdir(parents=True,exist_ok=True)
    x,num,cat=features_for_mode(df,mode)
    tr,va,te=chronological_split(df)
    dev=tr|va
    folds=dev_cv_folds(df,dev)
    y=df.checkins.to_numpy(float)
    study=tune(x,df,cat,folds,trials,SEED)
    params=study.best_params
    strategies=['baseline_median','baseline_same_type','catboost_fixed','catboost_optuna','weighted','global_adjusted','log_target']
    cv_rows=[]
    for strategy in strategies:
        curparams={} if strategy=='catboost_fixed' else params
        cv_rows.extend(cv_strategy(x,df,cat,folds,curparams,strategy))
    cv=pd.DataFrame(cv_rows)
    cv.to_csv(out/'validacion_temporal.csv',index=False,encoding='utf-8-sig')
    agg=cv.groupby('strategy',as_index=False).agg(cv_mae=('MAE','mean'),cv_mae_sd=('MAE','std'),
        cv_rmse=('RMSE','mean'),cv_global_main_n=('global_main_n','sum'),cv_global_main_mae=('global_main_mae','mean'))
    agg=agg.sort_values('cv_mae')
    agg.to_csv(out/'comparacion_validacion.csv',index=False,encoding='utf-8-sig')
    # Selección usando SOLO CV; una vez fijado el candidato, se evalúa test UNA vez.
    winner=str(agg.iloc[0]['strategy'])
    candidates=['catboost_fixed','catboost_optuna','weighted','global_adjusted','log_target']
    selected_model=next((s for s in agg.strategy if s in candidates),'catboost_fixed')
    eval_strategies=['baseline_median','baseline_same_type','catboost_fixed',selected_model]
    test=df.loc[te,['event_id','fecha_puebla','nombre_campfire','tipo_campfire','event_family','event_phase','checkins']].copy()
    test.rename(columns={'checkins':'real'},inplace=True)
    testr=[]; trained={}
    for strategy in dict.fromkeys(eval_strategies):
        if strategy=='baseline_median': p=np.full(int(te.sum()),np.median(y[dev]))
        elif strategy=='baseline_same_type':
            p=df.loc[te,'hist_same_type_last_5_median'].to_numpy(float)
            p[~np.isfinite(p)]=np.median(y[dev])
        else:
            actual_params={} if strategy=='catboost_fixed' else params
            weighted=strategy=='weighted';log_target=strategy=='log_target'
            model,p=fit_predict(x,y,dev,te,cat,actual_params,
                 weight_mode='high_attendance' if weighted else 'none',log_target=log_target)
            if strategy=='global_adjusted':
                p,_=specialized_predictions(model,x,y,dev,te,df,p)
            trained[strategy]=model
            model.save_model(str(out/f'{strategy}.cbm'))
        test['pred_'+strategy]=p
        m=evaluate(y[te],p)
        gm=(df.loc[te,'event_phase']=='main_event').to_numpy()
        m.update({'strategy':strategy,'global_main_n':int(gm.sum()),
              'global_main_mae':round(float(mean_absolute_error(y[te][gm],p[gm])),4) if gm.any() else None})
        testr.append(m)
    test.to_csv(out/'predicciones_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(testr).to_csv(out/'metricas_test.csv',index=False,encoding='utf-8-sig')
    bykind=[]
    for label,sub in test.groupby('event_phase'):
        for strategy in dict.fromkeys(eval_strategies):
            m=evaluate(sub.real,sub['pred_'+strategy]);m.update({'group':label,'n':len(sub),'strategy':strategy});bykind.append(m)
    pd.DataFrame(bykind).to_csv(out/'metricas_por_fase.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(study.trials_dataframe()).to_csv(out/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    best=model_params(params)
    (out/'optuna_mejores_parametros.json').write_text(json.dumps(best,ensure_ascii=False,indent=2),encoding='utf-8')
    report={'modo':mode,'n_total':len(df),'dev_rows':int(dev.sum()),'test_rows':int(te.sum()),
       'test_from':str(df.loc[te,'date'].min().date()),'test_to':str(df.loc[te,'date'].max().date()),
       'folds':[{'train_n':int(t.sum()),'validation_n':int(v.sum()),
                 'validation_from':str(df.loc[v,'date'].min().date()),
                 'validation_to':str(df.loc[v,'date'].max().date())} for t,v in folds],
       'best_cv_strategy_any':winner,'selected_trainable_strategy':selected_model,
       'optuna_trials':trials,'best_params':study.best_params,
       'caveats':['Se respeta test cronológico; no se usa para seleccionar modelos',
                  'Los agregados previos usan resultados de días anteriores: evaluación secuencial, no pronóstico de toda la temporada desde su inicio',
                  'Agrupación conservadora de días de la misma edición global',
                  'Ranking 2026 solo escenario retrospectivo, no refleja historia real',
                  'Corrección por familia usa residuales in-sample con suavizado; evaluar con escepticismo',
                  'Con pocas ediciones globales, la métrica específica es inestable']}
    (out/'reporte.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    print('\n===',mode.upper(),'===')
    print('CV:',agg[['strategy','cv_mae','cv_global_main_n']].to_string(index=False))
    print('TEST:',pd.DataFrame(testr)[['strategy','MAE','RMSE','global_main_mae']].to_string(index=False))


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--csv',default='dataset_maestro_548.csv')
    ap.add_argument('--mode',choices=['anticipado','retrospectivo','ambos'],default='ambos')
    ap.add_argument('--trials',type=int,default=25)
    ap.add_argument('--out',default='resultados_etapa3')
    a=ap.parse_args()
    if a.trials<1:ap.error('--trials debe ser >=1')
    raw=pd.read_csv(a.csv,low_memory=False)
    df=build_features(raw)
    # reconstruir la fase usando fases_eventos.py, nunca la etiqueta antigua de archivo.
    for mode in (['anticipado','retrospectivo'] if a.mode=='ambos' else [a.mode]):
        run_mode(df.copy(),mode,Path(a.out)/mode,a.trials)

if __name__=='__main__':main()
