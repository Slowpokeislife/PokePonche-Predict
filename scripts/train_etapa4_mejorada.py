"""PokePonche Predict: Etapa 4.2 — actividad de usuarios, CatBoost, RSVP.

MODOS:
* anticipado: evaluacion predictiva valida (datos historicos disponibles antes de fecha).
* rsvp_proxy_final: experimento retrospectivo SOLAMENTE; total RSVP final incluye
  CHECKED_IN del evento y NO ES una instantanea al inicio. Su metrica NO estima
  el error de un predictor operativo de inicio.

No se usan rsvp_status ni nombres personales como atributos individuales en el entrenamiento;
solo agregaciones historicas del comportamiento hasta el dia anterior.
"""
from __future__ import annotations
import argparse, json, warnings
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

SEED = 42
CATS = ['tipo_campfire','event_family','event_phase','major_event_subtype']
NUMS = ['year_offset','month','weekday','day_of_year','is_weekend',
        'is_global_major_event','is_global_pre_event','days_until_main_event',
        'events_prior','hist_last5','hist_last10','same_type_last3','same_type_last5',
        'same_type_prev_count','user_active_30d','user_active_90d','user_active_180d',
        'user_checkins_30d','user_checkins_90d','user_retention_90d',
        'users_with_3plus_prior_checkins','community_users_seen_prior',
        'days_since_last_event','prior_conversion_10','prior_conversion_same_type',
        'last_global_event_checkins','days_since_last_major_event','recent_accepted_mean_5']


def load_features(master_path:Path, people_path:Path):
    df=pd.read_csv(master_path,encoding='utf-8-sig',low_memory=False,dtype={'event_id':str})
    pe=pd.read_csv(people_path,encoding='utf-8-sig',low_memory=False,
                   usecols=['event_id','user_id','rsvp_status'],dtype=str)
    for name in ['event_id','fecha_puebla','checkins','tipo_campfire']:
        if name not in df: raise ValueError(f'Falta {name} en dataset maestro')
    df['date']=pd.to_datetime(df.fecha_puebla,format='%d/%m/%Y',errors='raise')
    df['checkins']=pd.to_numeric(df.checkins,errors='raise').astype(int)
    if len(df)!=548 or df.event_id.duplicated().any():
        raise ValueError('El maestro debe tener 548 event_id diferentes')
    if set(pe.event_id)-set(df.event_id):raise ValueError('event_id de usuarios no aparece en maestro')
    valid=pe[pe.rsvp_status.eq('CHECKED_IN')]
    check=valid.groupby('event_id').size().reindex(df.event_id,fill_value=0).to_numpy()
    if not np.array_equal(check,df.checkins.to_numpy()):
        raise ValueError('Los CHECKED_IN no coinciden evento por evento con el maestro')
    if valid.duplicated(['event_id','user_id']).any():
        raise ValueError('Existen check-ins duplicados por evento/usuario')
    a=pe[pe.rsvp_status.eq('ACCEPTED')].groupby('event_id').size()
    df['accepted_remaining_final']=df.event_id.map(a).fillna(0).astype(int)
    # Solo para experimento proxy. Incluye el objetivo checkins => fuga si se afirma pronostico.
    df['accepted_total_final_proxy']=df.accepted_remaining_final+df.checkins
    df=df.sort_values(['date','event_id']).reset_index(drop=True)
    df['year_offset']=df.date.dt.year-df.date.dt.year.min()
    df['month']=df.date.dt.month
    df['weekday']=df.date.dt.dayofweek
    df['day_of_year']=df.date.dt.dayofyear
    df['is_weekend']=(df.weekday>=5).astype(int)
    for c in CATS:
        if c not in df:df[c]='desconocido'
    for c in ['is_global_major_event','is_global_pre_event','days_until_main_event']:
        if c not in df:df[c]=np.nan
    byid={e:set(g.user_id.dropna()) for e,g in valid.groupby('event_id')}
    earlier=[] # (dt, users, checkins, accepted_total, event_type, major_family)
    history_users=defaultdict(int)
    feats={}
    last_date=None
    last_major=None
    for date,group in df.groupby('date',sort=True):
        date=pd.Timestamp(date)
        r30=[z for z in earlier if 0<(date-z[0]).days<=30]
        r90=[z for z in earlier if 0<(date-z[0]).days<=90]
        r180=[z for z in earlier if 0<(date-z[0]).days<=180]
        u30=set().union(*(z[1] for z in r30)) if r30 else set()
        u90=set().union(*(z[1] for z in r90)) if r90 else set()
        u180=set().union(*(z[1] for z in r180)) if r180 else set()
        ys=[z[2] for z in earlier]
        conversion=[z[2]/z[3] for z in earlier[-10:] if z[3]>0]
        for r in group.itertuples(index=False):
            ty=str(r.tipo_campfire)
            same=[z for z in earlier if z[4]==ty]
            sr=[z[2]/z[3] for z in same[-10:] if z[3]>0]
            feats[r.event_id]={
                'events_prior':len(earlier),
                'hist_last5':np.mean(ys[-5:]) if ys else np.nan,
                'hist_last10':np.mean(ys[-10:]) if ys else np.nan,
                'same_type_last3':np.mean([z[2] for z in same[-3:]]) if same else np.nan,
                'same_type_last5':np.mean([z[2] for z in same[-5:]]) if same else np.nan,
                'same_type_prev_count':len(same),
                'user_active_30d':len(u30),'user_active_90d':len(u90),'user_active_180d':len(u180),
                'user_checkins_30d':sum(len(z[1]) for z in r30),
                'user_checkins_90d':sum(len(z[1]) for z in r90),
                'user_retention_90d':len(u30)/len(u90) if u90 else np.nan,
                'users_with_3plus_prior_checkins':sum(history_users[u]>=3 for u in u90),
                'community_users_seen_prior':len(history_users),
                'days_since_last_event':(date-last_date).days if last_date is not None else np.nan,
                'prior_conversion_10':np.mean(conversion) if conversion else np.nan,
                'prior_conversion_same_type':np.mean(sr) if sr else np.nan,
                'recent_accepted_mean_5':np.mean([z[3] for z in earlier[-5:]]) if earlier else np.nan,
                'last_global_event_checkins':last_major[1] if last_major else np.nan,
                'days_since_last_major_event':(date-last_major[0]).days if last_major else np.nan,
            }
        # Actualizar historiales DESPUES de calcular todas las filas del dia.
        for r in group.itertuples(index=False):
            users=byid.get(r.event_id,set())
            earlier.append((date,users,int(r.checkins),int(r.accepted_total_final_proxy),str(r.tipo_campfire),str(r.event_family)))
            for u in users:history_users[u]+=1
            if str(r.event_phase)=='main_event' and int(r.is_global_major_event or 0)==1:
                last_major=(date,int(r.checkins))
        last_date=date
    f=pd.DataFrame.from_dict(feats,orient='index').reindex(df.event_id).reset_index(drop=True)
    return pd.concat([df,f],axis=1)


def group_temporal_split(df):
    # Corta por dias unicos; todos los registros del mismo dia quedan juntos.
    dates=np.array(sorted(df.date.unique()))
    n=len(dates)
    ia,ib=int(n*.70),int(n*.85)
    cuts=(pd.Timestamp(dates[ia]),pd.Timestamp(dates[ib]))
    tr=(df.date<cuts[0]).to_numpy(); va=((df.date>=cuts[0])&(df.date<cuts[1])).to_numpy()
    te=(df.date>=cuts[1]).to_numpy()
    if not (tr.any() and va.any() and te.any()):raise ValueError('Particion temporal vacia')
    return tr,va,te,cuts


def matrix(df,mode):
    nums=NUMS.copy()
    if mode=='rsvp_proxy_final':nums+=['accepted_total_final_proxy']
    cats=[c for c in CATS if c in df.columns]
    num=[c for c in nums if c in df.columns]
    x=df[num+cats].copy()
    for c in num:x[c]=pd.to_numeric(x[c],errors='coerce').fillna(-9999.0).astype(float)
    for c in cats:x[c]=x[c].fillna('desconocido').astype(str)
    return x,cats


def eval_metrics(y,p,threshold=100):
    y=np.asarray(y,float);p=np.maximum(np.asarray(p,float),0)
    err=np.abs(y-p);large=y>=threshold; tol=np.where(large,15,5)
    return {'n':int(len(y)), 'mae':float(err.mean()),'rmse':float(np.sqrt(np.mean((y-p)**2))),
        'wape_pct':float(err.sum()/max(1,y.sum())*100),
        'within_target_pct':float(np.mean(err<=tol)*100),
        'within_5_small_pct':float(np.mean(err[~large]<=5)*100) if (~large).any() else None,
        'within_15_large_pct':float(np.mean(err[large]<=15)*100) if large.any() else None,
        'small_n':int((~large).sum()),'large_n':int(large.sum()),'bias':float(np.mean(p-y))}


def train_cat(x,cats,y,mask,variant,iterations):
    options={'loss_function':'MAE','iterations':iterations,'depth':5,'learning_rate':0.045,
        'l2_leaf_reg':8,'verbose':False,'thread_count':2,'random_seed':SEED,
        'allow_writing_files':False}
    if variant=='poisson':options['loss_function']='Poisson'
    if variant=='shallow':options['depth']=3;options['l2_leaf_reg']=12
    if variant=='weighted':
        weights=1+np.minimum(y[mask]/100,3)
    else:weights=None
    model=CatBoostRegressor(**options)
    model.fit(x.loc[mask],y[mask],cat_features=cats,sample_weight=weights)
    return model


def historical_reference(df, mask):
    # Feature 'same_type_last5' ha sido calculada SOLO con fechas anteriores.
    return np.clip(df.loc[mask,'same_type_last5'].fillna(df.loc[mask,'hist_last5']).fillna(28).to_numpy(float),0,None)


def conversion_reference(df,mask):
    # Se puede evaluar SOLO como proxy final; no es pronostico al inicio real.
    a=df.loc[mask,'accepted_total_final_proxy'].to_numpy(float)
    conv=df.loc[mask,'prior_conversion_same_type'].fillna(df.loc[mask,'prior_conversion_10']).fillna(0.65).to_numpy(float)
    return a*np.clip(conv,0.05,0.98)


def run(root,output,iterations=500,threshold=100):
    root=Path(root).resolve(); out=root/output; out.mkdir(parents=True,exist_ok=True)
    df=load_features(root/'dataset_maestro_548.csv',root/'registros_usuarios_original.csv')
    tr,va,te,cuts=group_temporal_split(df)
    y=df.checkins.to_numpy(float)
    useful=['event_id','fecha_puebla','checkins','accepted_remaining_final','accepted_total_final_proxy','event_family','event_phase']+NUMS
    df[useful].to_csv(out/'features_historicas.csv',index=False,encoding='utf-8-sig')
    summary={ 'dataset':{'rows':len(df),'checkins':int(y.sum()),'train':int(tr.sum()),'validation':int(va.sum()),
               'test':int(te.sum()),'test_date_from':str(cuts[1].date())},
               'IMPORTANT':'Los resultados de rsvp_proxy_final incluyen fuga de informacion temporal: el conteo final incluye CHECKED_IN. NO representan prediccion operativa.', 'modes':{}}
    for mode in ['anticipado','rsvp_proxy_final']:
        x,cats=matrix(df,mode)
        folder=out/mode;folder.mkdir(exist_ok=True)
        variants=['base','shallow','weighted','poisson']
        records=[]
        for variant in variants:
            mod=train_cat(x,cats,y,tr,variant,iterations)
            pred=np.maximum(mod.predict(x.loc[va]),0)
            met=eval_metrics(y[va],pred,threshold)
            records.append({'variant':variant,'validation_mae':met['mae'],
                            'validation_within_target_pct':met['within_target_pct'],
                            'validation_wape_pct':met['wape_pct']})
        ref=historical_reference(df,va) if mode=='anticipado' else conversion_reference(df,va)
        met=eval_metrics(y[va],ref,threshold)
        records.append({'variant':'historical_baseline' if mode=='anticipado' else 'conversion_baseline',
                        'validation_mae':met['mae'],'validation_within_target_pct':met['within_target_pct'],
                        'validation_wape_pct':met['wape_pct']})
        tab=pd.DataFrame(records).sort_values(['validation_mae','validation_within_target_pct'],ascending=[True,False])
        tab.to_csv(folder/'seleccion_validacion.csv',index=False,encoding='utf-8-sig')
        winner=str(tab.iloc[0]['variant'])
        # Seleccion sin consultar test final.
        if winner.endswith('_baseline'):
            test_pred=historical_reference(df,te) if mode=='anticipado' else conversion_reference(df,te)
        else:
            selected=train_cat(x,cats,y,tr|va,winner,iterations)
            test_pred=np.maximum(selected.predict(x.loc[te]),0)
            selected.save_model(str(folder/'modelo_catboost.cbm'))
            pd.DataFrame({'variable':x.columns,'importancia':selected.get_feature_importance()}).sort_values('importancia',ascending=False).to_csv(folder/'importancias.csv',index=False,encoding='utf-8-sig')
        evalm=eval_metrics(y[te],test_pred,threshold)
        preds=df.loc[te,['event_id','fecha_puebla','nombre_campfire','tipo_campfire','event_family','event_phase','checkins','accepted_remaining_final','accepted_total_final_proxy']].copy()
        preds['prediccion']=np.round(test_pred,2)
        preds['error_absoluto']=np.abs(test_pred-y[te])
        preds['margen_objetivo']=np.where(y[te]>=threshold,15,5)
        preds['dentro_margen']=preds.error_absoluto<=preds.margen_objetivo
        preds.to_csv(folder/'predicciones_test.csv',index=False,encoding='utf-8-sig')
        preds.groupby('tipo_campfire',dropna=False).agg(n=('event_id','count'),mae=('error_absoluto','mean'),
           dentro_margen_pct=('dentro_margen',lambda s:100*s.mean())).to_csv(folder/'metricas_por_tipo.csv',encoding='utf-8-sig')
        report={'mode':mode,'winner_on_validation':winner,'metricas_test':evalm,'test_is_genuine_forecast_evaluation':mode=='anticipado',
                'caution': 'El RSVP final es igual a ACCEPTED restante + CHECKED_IN, por eso contiene el objetivo: diagnostico hindsight exclusivamente.' if mode!='anticipado' else 'Historicos de participacion calculados con dias anteriores exclusivamente.'}
        (folder/'reporte.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
        summary['modes'][mode]=report
    (out/'resumen.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(summary,indent=2,ensure_ascii=False))
    return summary

if __name__=='__main__':
    pa=argparse.ArgumentParser(description='PokePonche Predict Etapa 4.2')
    pa.add_argument('--root',default='.')
    pa.add_argument('--output',default='resultados_etapa4_mejorada')
    pa.add_argument('--iterations',type=int,default=500)
    pa.add_argument('--large-threshold',type=int,default=100)
    args=pa.parse_args()
    run(args.root,args.output,args.iterations,args.large_threshold)
