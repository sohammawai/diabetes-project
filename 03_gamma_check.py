import pandas as pd, numpy as np

toy = pd.DataFrame({
    "HighBP":   [1, 1, 1, 0, 0, 0],
    "Smoker":   [1, 1, 0, 0, 1, 1],
    "Diabetes": [1, 1, 0, 0, 1, 0],
})

def gamma_vprs(df, B, target, beta=1.0):
    if not B:
        return 0.0
    g = df.groupby(list(B))[target].agg(['size', 'sum'])
    p_pos = g['sum'] / g['size']
    purity = np.maximum(p_pos, 1 - p_pos)
    return g['size'][purity >= beta].sum() / len(df)

B = ["HighBP", "Smoker"]
print(round(gamma_vprs(toy, B, "Diabetes", 1.0), 4))   # expect 0.6667
print(round(gamma_vprs(toy, B, "Diabetes", 0.6), 4))   # expect 0.6667
print(round(gamma_vprs(toy, B, "Diabetes", 0.5), 4))   # expect 1.0