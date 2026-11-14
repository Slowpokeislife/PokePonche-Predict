"""Clasificación conservadora de eventos globales y eventos preparatorios.

La fecha de Campfire es la autoridad (ya validada manualmente).
No se crean filas para eventos que aún no ocurren.
"""
from __future__ import annotations
import re
import unicodedata
import pandas as pd

# Overrides de eventos que el organizador ha aclarado personalmente.
PRE_EVENT_IDS = {
    'EVT0356': 'wild_area', # 2026-11-09, Zacian
    'EVT0320': 'wild_area', # 2026-11-10, Zamazenta
}
# Día principal del Wild Area 2026 informado por la comunidad: fuera del dataset histórico.
MAIN_WILD_AREA_2026 = (pd.Timestamp('2026-11-14'), pd.Timestamp('2026-11-15'))


def norm(value: object) -> str:
    if pd.isna(value):return ''
    s=unicodedata.normalize('NFKD',str(value))
    return ''.join(ch for ch in s if not unicodedata.combining(ch)).lower()


def get_family(name: object, event_type: object, major_type: object = '') -> str:
    s=norm(name) + ' ' + norm(event_type) + ' ' + norm(major_type)
    if re.search(r'go\s*fest',s):return 'go_fest'
    if re.search(r'go\s*tour|tour\s*de\s*pokemon\s*go',s):return 'go_tour'
    if re.search(r'wild\s*area|area\s*silvestre|expedicion\s*silvestre',s):return 'wild_area'
    return 'no_global'


def phase_info(event_id: object, date: object, name: object, event_type: object, major_type: object = '') -> dict:
    date=pd.Timestamp(date)
    family=get_family(name,event_type,major_type)
    sid=str(event_id)
    name_norm=norm(name)
    if sid in PRE_EVENT_IDS:
        family=PRE_EVENT_IDS[sid]
        phase='pre_event'
        days_until_main=(MAIN_WILD_AREA_2026[0]-date).days
        basis='manual_confirmed_2026_wild_area_pre_event'
    elif family=='no_global':
        phase='not_global'
        days_until_main=None
        basis='no_global_family_detected'
    elif re.search(r'\b(into\s+the|pre[ -]?evento|preparativ|antes\s+del|camino\s+al|rumbo\s+al|semana\s+previa)\b',name_norm):
        phase='pre_event'
        days_until_main=None
        basis='pre_event_wording_review_date'
    else:
        # Afirmamos principal solo si hay evidencia explícita de familia en ficha/tipo.
        # Si su nombre menciona Go Fest/Tour/Wild Area, pero no es un evento preparatorio,
        # el clasificador lo toma como principal (hipótesis corregible).
        phase='main_event'
        days_until_main=0
        basis='named_event_default_main'
    return {'event_family':family,'event_phase':phase,
            'is_global_major_event':int(phase=='main_event'),
            'is_global_pre_event':int(phase=='pre_event'),
            'days_until_main_event':days_until_main,
            'event_phase_basis':basis}


def add_phases(df: pd.DataFrame) -> pd.DataFrame:
    out=df.copy()
    if 'date' in out:dates=pd.to_datetime(out['date'])
    else:dates=pd.to_datetime(out['fecha_puebla'],format='%d/%m/%Y')
    fields=[phase_info(row.get('event_id'),date,row.get('nombre_campfire',''),row.get('tipo_campfire',''),row.get('major_event_type',''))
            for (_,row),date in zip(out.iterrows(),dates)]
    x=pd.DataFrame(fields,index=out.index)
    for col in x.columns:out[col]=x[col]
    out['major_event_subtype']=out['event_family']
    return out
