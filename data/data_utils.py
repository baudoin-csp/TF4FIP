# %% 
# Helper function to compute the TP, FP, TN, FN for every row of one dataframe

# %%
import pandas as pd
import matplotlib.pyplot as plt
import ast
import numpy as np

# %%
models = {0: "moirai", 1: "chronos", 2: "time_moe"}

context_length = 384
context_length_index = context_length - 1
prediction_length = 3

# %%
def helper_metric(filepath: str, model: str):

    df = pd.read_csv(filepath, parse_dates=True, index_col=0)

    # TP: True Positive, TN: True Negative, FP: False Positive, FN: False Negative
    df["TP"] = 0
    df["TN"] = 0
    df["FP"] = 0
    df["FN"] = 0

    # We take the first 384 rows as context. We start predicting from the 385th row
    first_predicted_row = context_length_index
    last_predicted_row = len(df) - prediction_length
    
    for index, row in df[first_predicted_row:last_predicted_row].iterrows():
        base_price = row['Close']
        future_predictions = ast.literal_eval(row["Result"])
        future_prices = df.loc[index:].iloc[1:prediction_length+1]["Close"].tolist() # get the next prediction_length values corresponding to the next prediction_length horizons
        real_difference_signs = [np.sign(price - base_price) for price in future_prices]
        predicted_difference_signs = [np.sign(prediction - base_price) for prediction in future_predictions]

        TP = [0 for _ in range(prediction_length)]
        TN = [0 for _ in range(prediction_length)]
        FP = [0 for _ in range(prediction_length)]
        FN = [0 for _ in range(prediction_length)]

        # Compute the TP, TN, FP, FN for each horizon
        for horizon_index, (real_difference_sign, predicted_difference_sign) in enumerate(zip(real_difference_signs, predicted_difference_signs)):
            if real_difference_sign == predicted_difference_sign and real_difference_sign == 1:
                TP[horizon_index] += 1
            elif real_difference_sign == predicted_difference_sign and real_difference_sign == -1:
                TN[horizon_index] += 1
            elif real_difference_sign != predicted_difference_sign and real_difference_sign == 1:
                FN[horizon_index] += 1
            elif real_difference_sign != predicted_difference_sign and real_difference_sign == -1:
                FP[horizon_index] += 1

        # fill the column for TP, TN, FP, FN for the current row
        df.at[index, "TP"] = str(TP)
        df.at[index, "TN"] = str(TN)
        df.at[index, "FP"] = str(FP)
        df.at[index, "FN"] = str(FN)
        
   
    df.to_csv(f"analysis/future_data/test/data_2023_2024_{model}.csv", index=True)

#Compute the TP, TN, FP, FN colulns for each model's dataframe
for model_index, model_name in models.items():
    helper_metric(filepath=f"analysis/future_data/test/data_2023_2024_{model_name}.csv", model=model_name)

# %%