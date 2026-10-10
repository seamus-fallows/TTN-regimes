import pandas as pd, glob, os
R = r"c:/Users/seamu/Iliad-ML/TTN_regimes/experiments/optim_samples/results"
for f in sorted(glob.glob(R + "/*.csv")):
    df = pd.read_csv(f)
    print(os.path.basename(f), df.shape)
    print("   ", list(df.columns))
