"""Etapa 5: etiquetas corregidas, historial de usuarios, evaluación al inicio (si hay snapshots reales).
NO utiliza el RSVP final como característica de un predictor operativo.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error
import optuna
from train_etapa2 import build_features, BASE_NUMERICS, RETRO_NUMERICS, chronological_split
from train_etapa3 import dev_cv_folds
from train_etapa4_mejorada import load_features, NUMS as USER_NUMS
from fases_eventos import PRE_EVENT_IDS

SEED=42
IDS=set(PRE_EVENT_IDS)
CATS=['tipo_campfire','event_phase','event_family','major_event_subtype','month_label']
USER_EXTRA=[c for c in USER_NUMS if c.startswith(('user_','users_','community_','prior_conversion','recent_accepted'))]


def prepare(master,people):
    raw=pd.read_csv(master,encoding='utf-8-sig',low_memory=False,dtype={'event_id':str})
    raw['tipo_campfire_original']=raw['tipo_campfire'].copy()
    # No alteramos maestro original en disco. Solo construimos etiqueta para modelado.
    is_pre=raw.event_id.isin(IDS)
    raw.loc[is_pre,'tipo_campfire']='wild_area_pre_event'
    a=build_features(raw)
    hist=load_features(Path(master),Path(people))
    cols=['event_id']+ [c for c in USER_EXTRA if c in hist]
    a=a.merge(hist[cols],on='event_id',how='left',validate='one_to_one')
    if len(a)!=548 or a.event_id.duplicated().any() or int(a.checkins.sum())!=23032:
        raise ValueError('Error de integridad 548 filas / 23032 checkins')
    for eid in IDS:
        r=a.loc[a.event_id==eid].iloc[0]
        assert r['tipo_campfire']=='wild_area_pre_event' and r['event_phase']=='pre_event'
    return a


def make_matrix(df,scenario='anticipado'):
    nums=list(dict.fromkeys(BASE_NUMERICS+USER_EXTRA+(RETRO_NUMERICS if scenario=='retrospectivo' else [])))
    nums=[c for c in nums if c in df]
    cats=[c for c in CATS if c in df]
    x=df[nums+cats].copy()
    for c in nums:x[c]=pd.to_numeric(x[c],errors='coerce').fillna(-99999).astype(float)
    for c in cats:x[c]=x[c].fillna('desconocido').astype(str)
    return x,cats


def metrics(y,p,threshold=100):
    y=np.asarray(y,float);p=np.maximum(np.asarray(p,float),0)
    d=np.abs(y-p);large=y>=threshold;tol=np.where(large,15,5)
    return {'n':int(len(y)),'MAE':round(float(d.mean()),4),'RMSE':round(float(np.sqrt(np.mean((y-p)**2))),4),
      'WAPE_pct':round(float(d.sum()/max(1,y.sum())*100),3),
      'dentro_margen_pct':round(float(np.mean(d<=tol)*100),3),
      'pequenos_dentro_5_pct':round(float(np.mean(d[~large]<=5)*100),3) if (~large).any() else None,
      'grandes_dentro_15_pct':round(float(np.mean(d[large]<=15)*100),3) if large.any() else None,
      'bias':round(float(np.mean(p-y)),3)}


def model_params(p=None):
    q=dict(iterations=450,depth=4,learning_rate=.04,l2_leaf_reg=8,random_strength=1.0,
           random_seed=SEED,loss_function='MAE',verbose=False,allow_writing_files=False,thread_count=2)
    if p:q.update(p)
    return q


def fit_pred(x,y,cat,tr,va,params,variant):
    target=np.log1p(y[tr]) if variant=='log' else y[tr]
    weights=None
    if variant=='weighted':
        q=np.quantile(y[tr],.80);weights=np.where(y[tr]>=q,1.7,1.)
    m=CatBoostRegressor(**model_params(params))
    m.fit(x.loc[tr],target,cat_features=cat,sample_weight=weights)
    pred=m.predict(x.loc[va]);pred=np.expm1(pred) if variant=='log' else pred
    return m,np.maximum(pred,0)


def baseline(df,mask,dev_y):
    a=df.loc[mask,'hist_same_type_last_5_median'].to_numpy(dtype=float)
    b=df.loc[mask,'hist_last_10_median'].to_numpy(dtype=float)
    a=np.where(np.isfinite(a),a,b)
    return np.where(np.isfinite(a),a,float(np.median(dev_y)))


def execute_mode(df,mode,out,trials,iterations):
    out.mkdir(parents=True,exist_ok=True)
    x,cats=make_matrix(df,mode);y=df.checkins.to_numpy(float)
    tr,va,te=chronological_split(df);dev=tr|va
    folds=dev_cv_folds(df,dev)
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    def obj(t):
        p={'iterations':t.suggest_int('iterations',220,iterations,step=50),
           'depth':t.suggest_int('depth',3,6),
           'learning_rate':t.suggest_float('learning_rate',.018,.11,log=True),
           'l2_leaf_reg':t.suggest_float('l2_leaf_reg',2,20,log=True)}
        scores=[]
        for aa,bb in folds:
            _,pred=fit_pred(x,y,cats,aa,bb,p,'raw')
            scores.append(mean_absolute_error(y[bb],pred))
        return float(np.mean(scores))
    study=optuna.create_study(direction='minimize',sampler=optuna.samplers.TPESampler(seed=SEED))
    study.optimize(obj,n_trials=trials,show_progress_bar=False)
    strategies=['baseline','raw','log','weighted']
    cv=[]
    for variant in strategies:
        for fold,(aa,bb) in enumerate(folds,1):
            if variant=='baseline':pred=baseline(df,bb,y[aa])
            else:_,pred=fit_pred(x,y,cats,aa,bb,study.best_params,variant)
            cv.append({'variant':variant,'fold':fold,**metrics(y[bb],pred)})
    val=pd.DataFrame(cv);val.to_csv(out/'validacion_temporal.csv',index=False,encoding='utf-8-sig')
    selection=val.groupby('variant',as_index=False).agg(MAE_CV=('MAE','mean'),dentro_margen_CV=('dentro_margen_pct','mean'))
    selection=selection.sort_values(['MAE_CV','dentro_margen_CV'],ascending=[True,False])
    selection.to_csv(out/'seleccion_validacion.csv',index=False,encoding='utf-8-sig')
    winner=str(selection.iloc[0]['variant'])
    if winner=='baseline':p=baseline(df,te,y[dev])
    else:
        model,p=fit_pred(x,y,cats,dev,te,study.best_params,winner)
        model.save_model(str(out/'modelo_catboost.cbm'))
        pd.DataFrame({'feature':x.columns,'importance':model.get_feature_importance()}).sort_values('importance',ascending=False).to_csv(out/'importancias.csv',index=False,encoding='utf-8-sig')
    test=df.loc[te,['event_id','fecha_puebla','nombre_campfire','tipo_campfire_original','tipo_campfire','event_family','event_phase','checkins']].copy()
    test['prediccion']=p;test['error_abs']=np.abs(y[te]-p);test['tolerancia']=np.where(y[te]>=100,15,5);test['dentro_margen']=test.error_abs<=test.tolerancia
    test.to_csv(out/'predicciones_test.csv',index=False,encoding='utf-8-sig')
    pd.DataFrame([{'variant':winner,**metrics(y[te],p)}]).to_csv(out/'metricas_test.csv',index=False,encoding='utf-8-sig')
    test.groupby('event_phase').agg(n=('event_id','count'),mae=('error_abs','mean'),dentro_pct=('dentro_margen','mean')).to_csv(out/'metricas_por_fase.csv',encoding='utf-8-sig')
    pd.DataFrame(study.trials_dataframe()).to_csv(out/'optuna_trials.csv',index=False,encoding='utf-8-sig')
    report={'escenario':mode,'seleccionado_por_validacion':winner,'test':metrics(y[te],p),'trials':trials,'best_params':study.best_params,
      'nota':'Las filas 9 y 10 nov 2026 cambian tipo DE ENTRENAMIENTO a wild_area_pre_event; el tipo de origen queda sin alterar. El test NO se usa para elegir modelo.',
      'advertencias':['Históricos previos calculados con resultados de días anteriores; evaluación secuencial',
      'Retrospectivo incluye datos ex-post/rankings actuales y no mide predicción operativa',
      'El contador RSVP FINAL no participa en el entrenamiento anticipado. Hace falta snapshot real al inicio.']}
    (out/'reporte.json').write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    return report


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--root',default='.')
    ap.add_argument('--trials',type=int,default=15)
    ap.add_argument('--iterations',type=int,default=520)
    ap.add_argument('--mode',choices=['anticipado','retrospectivo','ambos'],default='ambos')
    ap.add_argument('--out',default='resultados_etapa5')
    args=ap.parse_args()
    if args.trials<1 or args.iterations<270:ap.error('--trials >=1 y --iterations >=270')
    root=Path(args.root);a=prepare(root/'data'/'dataset_maestro_548.csv',root/'data'/'registros_usuarios_pseudonimizados.csv')
    out=root/args.out;out.mkdir(exist_ok=True,parents=True)
    a[['event_id','fecha_puebla','tipo_campfire_original','tipo_campfire','event_family','event_phase','checkins']].to_csv(out/'auditoria_etiquetas.csv',index=False,encoding='utf-8-sig')
    modes=['anticipado','retrospectivo'] if args.mode=='ambos' else [args.mode]
    reps={m:execute_mode(a,m,out/m,args.trials,args.iterations) for m in modes}
    (out/'resumen.json').write_text(json.dumps(reps,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({m:r['test'] for m,r in reps.items()},indent=2,ensure_ascii=False))

if __name__=='__main__':main()
