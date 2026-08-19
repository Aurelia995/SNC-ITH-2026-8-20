import hashlib
from pathlib import Path
import pandas as pd

d=pd.read_csv('/root/autodl-tmp/HCR_runs/reproducibility_20260820/Table_L1_Patient_Dice.csv')
print('exact_DSC1',sum(abs(d.DSC-1)<1e-12),'DSC_lt_0.9',sum(d.DSC<.9),'DSC_zero',sum(d.DSC==0))
print(d.sort_values('DSC')[['patient_id','DSC','geometry_originally_matched','mask2_resampled_to_mask1','mask1_volume_cm3','mask2_volume_cm3','Center']].head(15).to_string(index=False))
p1=Path('/root/autodl-tmp/HCR_input/Icc1/mask1'); p2=Path('/root/autodl-tmp/HCR_input/Icc2/mask2')
rows=[]
for a in sorted(p1.glob('*.nii.gz')):
 b=p2/a.name
 h1=hashlib.md5(a.read_bytes()).hexdigest(); h2=hashlib.md5(b.read_bytes()).hexdigest()
 rows.append({'file':a.name,'md5_1':h1,'md5_2':h2,'binary_identical':h1==h2})
m=pd.DataFrame(rows); print('binary_identical',m.binary_identical.sum(),'of',len(m)); print(m.to_string(index=False)); m.to_csv('/root/autodl-tmp/HCR_runs/reproducibility_20260820/Mask_pair_MD5_audit.csv',index=False)
