"""Etapa 5 RSVP: entrena usando el total visible de interesados.

Reconstruccion del contador: ACCEPTED + CHECKED_IN en el CSV.
No existe marca de tiempo historica del contador: la evaluacion es retrospectiva
con proxy de contador visible, NO prueba que ya estuviera completo al inicio.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
import optuna
from train_etapa5 import prepare, make_matrix, chronological_split, dev_cv_folds, metrics, model_params

SEED=42
VALID_STATUS={'ACCEPTED','CHECKED_IN'}

def add_rsvp(df, people_path):
    pe=pd.read_csv(people_path,encoding='utf-8-sig',usecols=['event_id','user_id','rsvp_status'],dtype=str)
    # Una persona cuenta solo una vez por quedada y estado final.
    if pe.duplicated(['event_id','user_id']).any():
        raise ValueError('Hay usuarios duplicados dentro del mismo evento')
    actual=pe[pe.rsvp_status.eq('CHECKED_IN')].groupby('event_id').size()
    accepted=pe[pe.rsvp_status.eq('ACCEPTED')].groupby('event_id').size()
    out=df.copy()
    out['interesados_total_proxy']=(out.event_id.map(actual).fillna(0).astype(int)
                                   +out.event_id.map(accepted).fillna(0).astype(int))
    out['accepted_remaining_final']=out.event_id.map(accepted).fillna(0).astype(int)
    if not (out.interesados_total_proxy>=out.checkins).all():
        raise ValueError('El total de interesados no puede ser menor que los check-ins')
    if not np.array_equal(out.event_id.map(actual).fillna(0).to_numpy(dtype=int),out.checkins.to_numpy(dtype=int)):
        raise ValueError('Los check-ins del maestro no coinciden con registros de usuarios')
    return out


def prior_ratios(df):
    # Historial acumulado hasta el DIA ANTERIOR, nunca de la misma fecha.
    result={}
    prev=[]
    for date,grp in df.groupby('date',sort=True):
        for row in grp.itertuples(index=False):
            same=[r for r in prev if r[0]==row.tipo_campfire]
            recent=same[-15:]
            global_recent=prev[-50:]
            # beta-binomial suavizada hacia conversion global, sin usar futuro.
            q=sum(z[2] for z in global_recent)/max(1,sum(z[1] for z in global_recent)) if global_recent else 0.70
            n=sum(z[1] for z in recent)
            successes=sum(z[2] for z in recent)
            alpha=20
            rate=(successes+alpha*q)/(n+alpha)
            result[row.event_id]=rate
        for row in grp.itertuples(index=False):
            prev.append((row.tipo_campfire,int(row.interesados_total_proxy),int(row.checkins)))
    return df.event_id.map(result).astype(float).to_numpy()


def fit_model(x,y,cats,tr,va,params,kind):
    model=CatBoostRegressor(**model_params(params))
    if kind=='conversion':
        # La tasa por evento puede ser 0..1. No es la tasa historica del mismo evento como feature.
        n=np.maximum(x.loc[tr,'interesados_total_proxy'].to_numpy(float),1)
        target=np.clip(y[tr]/n,0,1)
        model.fit(x.loc[tr].drop(columns=['interesados_total_proxy']),target,cat_features=cats)
        pred=np.clip(model.predict(x.loc[va].drop(columns=['interesados_total_proxy'])),0,1)*x.loc[va,'interesados_total_proxy'].to_numpy(float)
    else:
        target=np.log1p(y[tr]) if kind=='log' else y[tr]
        model.fit(x.loc[tr],target,cat_features=cats)
        pred=model.predict(x.loc[va])
        if kind=='log': pred=np.expm1(pred)
    return model,np.maximum(pred,0)


def main():
    ap=argparse.ArgumentParser(description='Entrena con interesados Campfire (= CHECKED_IN + ACCEPTED).')
    ap.add_argument('--root',default='.')
    ap.add_argument('--out',default='resultados_etapa5_rsvp')
    ap.add_argument('--trials',type=int,default=12)
    args=ap.parse_args()
    root=Path(args.root);out=root/args.out;out.mkdir(parents=True,exist_ok=True)
    data=add_rsvp(prepare(root/'data'/'dataset_maestro_548.csv',root/'data'/'registros_usuarios_pseudonimizados.csv'),root/'data'/'registros_usuarios_pseudonimizados.csv')
    x,cats=make_matrix(data,'anticipado')
    # Se añade contador de interesados a modelo directo, no estados individuales.
    x['interesados_total_proxy']=data.interesados_total_proxy.to_numpy(float)
    x['conversion_historica_anterior']=prior_ratios(data)
    y=data.checkins.to_numpy(float)
    tr,va,te=chronological_split(data);dev=tr|va
    folds=dev_cv_folds(data,dev)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def objective(t):
        p={'iterations':t.suggest_int('iterations',200,550,step=50),
           'depth':t.suggest_int('depth',3,6),
           'learning_rate':t.suggest_float('learning_rate',.02,.12,log=True),
           'l2_leaf_reg':t.suggest_float('l2_leaf_reg',2,20,log=True)}
        scores=[]
        for a,b in folds:
            _,pred=fit_model(x,y,cats,a,b,p,'direct')
            scores.append(metrics(y[b],pred)['MAE'])
        return np.mean(scores)
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(objective,n_trials=args.trials,show_progress_bar=False)
    opts=['historical_conversion','direct','log','conversion']
    cv=[]
    for name in opts:
        for fold,(a,b) in enumerate(folds,1):
            if name=='historical_conversion':
                pred=x.loc[b,'interesados_total_proxy'].to_numpy(float)*x.loc[b,'conversion_historica_anterior'].to_numpy(float)
            else: _,pred=fit_model(x,y,cats,a,b,study.best_params,name)
            cv.append({'model':name,'fold':fold,**metrics(y[b],pred)})
    cv=pd.DataFrame(cv);cv.to_csv(out/'validacion_temporal.csv',index=False,encoding='utf-8-sig')
    selection=cv.groupby('model',as_index=False).agg(MAE_CV=('MAE','mean'),dentro_margen_CV=('dentro_margen_pct','mean')).sort_values(['MAE_CV','dentro_margen_CV'],ascending=[True,False])
    selection.to_csv(out/'seleccion_modelo.csv',index=False,encoding='utf-8-sig')
    winner=str(selection.iloc[0]['model'])
    if winner=='historical_conversion':
        p=x.loc[te,'interesados_total_proxy'].to_numpy(float)*x.loc[te,'conversion_historica_anterior'].to_numpy(float)
    else:
        model,p=fit_model(x,y,cats,dev,te,study.best_params,winner)
        model.save_model(str(out/'modelo_catboost.cbm'))
    test=data.loc[te,['event_id','fecha_puebla','nombre_campfire','tipo_campfire','event_phase','checkins','interesados_total_proxy','accepted_remaining_final']].copy()
    test['prediccion']=np.round(p,2);test['error_abs']=np.abs(y[te]-p)
    test['tolerancia']=np.where(y[te]>=100,15,5)
    test['dentro_margen']=test.error_abs<=test.tolerancia
    test.to_csv(out/'predicciones_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'model':winner,**metrics(y[te],p)}]).to_csv(out/'metricas_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame(study.trials_dataframe()).to_csv(out/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    report={'eventos':len(data),'checkins':int(data.checkins.sum()),'rsvp_columna':'interesados_total_proxy = CHECKED_IN + ACCEPTED',
      'ganador_validacion':winner,'metricas_test':metrics(y[te],p),'trials':args.trials,'mejores_parametros':study.best_params,
      'limitacion':'El CSV registra estados finales sin timestamp. Esta evaluacion usa el total reconstruido como proxy de contador visible. No demuestra que ese total se conociera al inicio del evento.',
      'uso_operativo':'En un evento futuro, usar el contador visible real de Voy a asistir al inicio y las mismas variables ex-ante; nunca usar los check-ins del evento para crear ese contador.'}
    (out/'reporte.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps({'ganador':winner,'test':report['metricas_test'],'carpeta':str(out)},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
