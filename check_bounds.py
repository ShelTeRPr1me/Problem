import pandas as pd
df = pd.read_csv(r'D:\WorkSpace\question\2026年XJTU校赛题目\2026年校赛题目\A\Cement_ESP_Data.csv')
for i in range(1,5):
    u = df[f'U{i}_kV']
    t = df[f'T{i}_s']
    print(f'U{i}: min={u.min():.1f}, max={u.max():.1f}, max*1.2={u.max()*1.2:.1f}')
    print(f'T{i}: min={t.min():.0f}, max={t.max():.0f}, median={t.median():.0f}, Q95={t.quantile(0.95):.0f}')