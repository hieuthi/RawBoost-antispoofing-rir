import sys
import pandas as pd

scorefile=sys.argv[1]
threshold=float(sys.argv[2])

metafile='C1R2_eval_metadata.csv'

dfscore = pd.read_csv(scorefile, sep=" ", header=None, names=["Utterance", "Score"])
dfmeta = pd.read_csv(metafile)
df = pd.merge(dfmeta, dfscore, how='left', on='Utterance')
df['Prediction'] = df['Score'] > threshold
dffake = df[df['RIR'].notnull()]
far = len(dffake[dffake['Prediction']]) / len(dffake)
print(f"False Acceptance Rate (overall): {far*100:.3f}%")

drr_mins = [-10, -5, 0, 5, 10, 15, 20, 25]
drr_maxs = [-5, 0, 5, 10, 15, 20, 25, 30]
t60_mins = [0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75]
t60_maxs = [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
ret = []
for i in range(len(drr_mins)):
  filtered_df = dffake.query(f'DRR>{drr_mins[i]} and DRR<={drr_maxs[i]}')
  # print(f'{drr_mins[i]} {drr_maxs[i]} {len(filtered_df)}')
  row = []
  for j in range(len(t60_mins)):
    focused_df = filtered_df.query(f'T60>{t60_mins[j]} and T60<={t60_maxs[j]}')
    success_df = focused_df.query('Prediction')
    count_all = len(focused_df)
    count_success = len(success_df)
    # print(f'  {t60_mins[j]} {t60_maxs[j]} {count_all} {count_success} {count_success/count_all}')
    row.append(count_success/count_all)
  ret.append(row)


for j in reversed(range(len(t60_mins))):
    print(f"{t60_mins[j]}\t", end="")
    for i in range(len(drr_mins)):
        print(f"{ret[i][j]*100:.02f}", end =" ")
    print("")

print(f"FAR\t{' '.join([str(x) for x in drr_mins])}")

