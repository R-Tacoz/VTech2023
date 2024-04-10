#%% Imports & Setup
import pandas as pd
from utils import to_clocktime

CSV_FILEPATH = "D:/Python/VTech/bysubredditdata/common_data_for_agreement_evaluation.csv"
ROW_PROGRESS_FILEPATH = "D:/Python/VTech/rowprogress.txt"
start_row = 345
#%% Main

#def labeling_session(start_row: int=-1):
df = pd.read_csv(CSV_FILEPATH, header=0, index_col=0)
label_count = 0
if start_row < 0:
    with open(ROW_PROGRESS_FILEPATH) as f:
        row_progress = int(f.readline().strip())
else:
    row_progress = start_row

idx = -1
for _, row in df.iterrows():
    idx += 1
    if idx < row_progress: continue
    
    print("\n------------------------------------")
    print(f"Row {idx} Title:", row['title'], f"({to_clocktime(row['created_utc'])})")
    print("Content:", row['selftext'])
    print("\nanxiety, bipolar, depression:", row['anxiety'], row['bipolar'], row['depression'])
    x = input("New label? ")
    if len(x) == 0: continue
    
    try:
        # 1 is negative, 2 is positive (the keys are closer than 1 and 0)
        newanx, newbip, newdep = int(x[0])-1, int(x[1])-1, int(x[2])-1
    except Exception as e:
        print(e, ", ending session")
        print("You labeled", label_count, "rows this session")
        while True:
            try:
                df.to_csv(CSV_FILEPATH)
                break
            except PermissionError:
                input("PermissionError, please close Excel before ending")
            
        if start_row < 0:
            with open(ROW_PROGRESS_FILEPATH, 'w') as f:
                f.write(str(row_progress + label_count))
        break
    
    df.loc[idx, 'anxiety'] = newanx
    df.loc[idx, 'bipolar'] = newbip
    df.loc[idx, 'depression'] = newdep
            
    label_count += 1

#if __name__=="__main__": labeling_session(start_row=start_row)