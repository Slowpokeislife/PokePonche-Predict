"""Verificación básica de integridad para el dataset público."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
m=pd.read_csv(ROOT/'data/dataset_maestro_548.csv',encoding='utf-8-sig')
p=pd.read_csv(ROOT/'data/registros_usuarios_pseudonimizados.csv',encoding='utf-8-sig')
assert len(m)==548 and len(p)==33848
assert not m.event_id.duplicated().any()
assert not p.duplicated(['event_id','user_id']).any()
assert p.user_id.str.fullmatch(r'USUARIO\d+').all()
assert p.username.equals(p.user_id)
assert set(p.event_id).issubset(set(m.event_id))
actual=p[p.rsvp_status.eq('CHECKED_IN')].groupby('event_id').size()
assert (m.event_id.map(actual).fillna(0).astype(int).to_numpy()==m.checkins.to_numpy()).all()
assert m.checkins.sum()==23032
print('OK: 548 quedadas, 33,848 respuestas, 23,032 check-ins y seudónimos consistentes')
