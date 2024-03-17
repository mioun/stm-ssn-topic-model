import os.path

import numpy as np
import pandas as pd

DATA_SET_PATH = f'model-output-data/ag-bert-stm-comparision'
df = pd.read_csv(os.path.join(DATA_SET_PATH, '_coherence-results.csv'))

print(df)
def calculate_metrics(metric):
    data = df.loc[(df['metric'] == metric) & (df['model'] != 'STM') ]
    avg_npmi = data['value'].mean()
    return np.round(avg_npmi, 3)



npmi = calculate_metrics('npmi')
ca = calculate_metrics('ca')
puw = calculate_metrics('PUW')

print(f' NPMI : {npmi}')
print(f' CA : {ca}')
print(f' PUW : {puw}')
print(f' TNPMI : {np.round(puw * npmi, 3)}')
print(f' TCA : {np.round(puw * ca, 3)}')
