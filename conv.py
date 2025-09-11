import wfdb
import pandas as pd

record = wfdb.rdrecord("bidmc02")  


print("Channels:", record.sig_name)

ppg_idx = record.sig_name.index("PLETH,")
ppg = record.p_signal[:, ppg_idx]


pd.DataFrame({"ppg": ppg}).to_csv("bidmc02_ppg.csv", index=False)
print("Saved bidmc02_ppg.csv")
