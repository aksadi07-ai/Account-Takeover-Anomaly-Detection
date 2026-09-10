"""Create session-level behavioural features and padded event sequences."""
from pathlib import Path
import json, numpy as np, pandas as pd
EVENTS=["LOGIN","ACCOUNT_VIEW","BALANCE_CHECK","BENEFICIARY_VIEW","BENEFICIARY_ADD","PASSWORD_CHANGE","SECURITY_CHANGE","DEVICE_REGISTER","TRANSFER","LOGOUT","FAILED_LOGIN"]
INP=Path('data/processed/sessionized_events.csv'); FEATURES=Path('data/processed/session_features.csv'); SEQUENCES=Path('data/processed/sequences.npz'); META=Path('data/processed/sequence_meta.json')

def haversine(lat1,lon1,lat2,lon2):
    r=6371.; p1,p2=np.radians(lat1),np.radians(lat2); dp=np.radians(lat2-lat1); dl=np.radians(lon2-lon1); a=np.sin(dp/2)**2+np.cos(p1)*np.cos(p2)*np.sin(dl/2)**2; return 2*r*np.arcsin(np.sqrt(a))

def main():
    df=pd.read_csv(INP,parse_dates=['timestamp']).sort_values(['user_id','timestamp']).copy()
    # Previous event for every raw event; used for session-to-session novelty/travel.
    df['prev_ts']=df.groupby('user_id').timestamp.shift(1); df['prev_lat']=df.groupby('user_id').latitude.shift(1); df['prev_lon']=df.groupby('user_id').longitude.shift(1)
    df['prev_device']=df.groupby('user_id').device.shift(1); df['prev_location']=df.groupby('user_id').location.shift(1)
    g=df.groupby('derived_session_id',sort=False)
    first=g.first(); last_ts=g.timestamp.max(); first_ts=g.timestamp.min()
    feat=pd.DataFrame(index=first.index)
    feat['user_id']=first.user_id; feat['label']=g.session_anomaly.max().astype(int); feat['attack_type']=g.session_attack_type.first()
    feat['event_count']=g.size(); feat['session_duration_sec']=(last_ts-first_ts).dt.total_seconds()
    for e,name in [('FAILED_LOGIN','failed_logins'),('BENEFICIARY_ADD','beneficiary_adds'),('PASSWORD_CHANGE','password_changes'),('SECURITY_CHANGE','security_changes'),('DEVICE_REGISTER','device_registers'),('TRANSFER','transfer_count')]:
        feat[name]=df.assign(v=(df.event==e)).groupby('derived_session_id').v.sum()
    tr=df[df.event=='TRANSFER']; amt=tr.groupby('derived_session_id').amount.agg(['sum','max','mean']).fillna(0)
    feat['total_transfer_amount']=amt['sum'].reindex(feat.index,fill_value=0); feat['max_transfer_amount']=amt['max'].reindex(feat.index,fill_value=0); feat['avg_transfer_amount']=amt['mean'].reindex(feat.index,fill_value=0)
    feat['transaction_velocity_per_min']=feat.transfer_count/np.maximum(feat.session_duration_sec/60,1)
    # First event novelty and impossible travel from the immediately preceding user event.
    feat['new_device']=(first.device != first.prev_device).astype(int).where(first.prev_device.notna(),0)
    feat['new_location']=(first.location != first.prev_location).astype(int).where(first.prev_location.notna(),0)
    hours=(first.timestamp-first.prev_ts).dt.total_seconds()/3600
    feat['time_since_prev_session_hr']=hours.fillna(999).clip(lower=0)
    dist=haversine(first.latitude,first.longitude,first.prev_lat,first.prev_lon)
    feat['impossible_travel']=((dist/np.maximum(hours.fillna(999),0.05))>800).astype(int).where(first.prev_ts.notna(),0)
    # User baseline uses only normal events, avoiding anomalous transfer amounts.
    normal=df[df.session_anomaly==0]; base=normal[normal.event=='TRANSFER'].groupby('user_id').amount.agg(['mean','std'])
    mean=feat.user_id.map(base['mean']).fillna(5000); std=feat.user_id.map(base['std']).fillna(3000).clip(lower=1)
    feat['amount_zscore']=(feat.avg_transfer_amount-mean)/std
    feat['login_hour']=first.timestamp.dt.hour
    feat=feat.reset_index().rename(columns={'derived_session_id':'session_id'})
    feat.to_csv(FEATURES,index=False)
    # Preserve chronological event order for sequence model.
    event_to_id={e:i+1 for i,e in enumerate(EVENTS)}; seqs=[]
    for _,x in df.groupby('derived_session_id',sort=False): seqs.append([event_to_id[e] for e in x.sort_values('timestamp').event])
    max_len=max(map(len,seqs)); padded=np.zeros((len(seqs),max_len),dtype=np.int64)
    for i,s in enumerate(seqs): padded[i,:len(s)]=s
    np.savez_compressed(SEQUENCES,sequences=padded,labels=feat.label.to_numpy())
    META.write_text(json.dumps({'event_to_id':event_to_id,'id_to_event':{str(i+1):e for i,e in enumerate(EVENTS)},'session_ids':feat.session_id.tolist(),'attack_types':feat.attack_type.tolist(),'max_len':max_len},indent=2))
    print(f'Wrote {len(feat):,} sessions; sequence shape={padded.shape}')
if __name__=='__main__': main()
