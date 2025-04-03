# chronos_prediction.py
import numpy as np
import torch
import json


def make_prediction(context_list, model, args):
    """
    Generate forecasts for a batch of context windows using Chronos.
    context_list: list of DataFrames, each representing one context window
    model: the loaded Chronos model
    args: script arguments (we need input_column, prediction_length, return_type, etc.)
    """
    

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and args.backend == "gpu") else "cpu"
    )

    # Prepare a list to store forecast outputs
    all_forecasts = []

    # For each window: standardize, keep track of mean and std for de-standardizing
    standardized_inputs = []
    means_stds = []
    for context_df in context_list:
        context = context_df[args.input_column].to_numpy()
        mean_val, std_val = context.mean(), context.std()
        # Handle edge case if std_val is 0 to avoid NaN
        if std_val == 0:
            std_val = 1e-8
        standardized = (context - mean_val) / std_val
        standardized_inputs.append(torch.tensor(standardized, dtype=torch.float32))
        means_stds.append((mean_val, std_val))

    # Stack into a single batch tensor
    # shape: (batch_size, context_length)
    batch_tensor = torch.stack(standardized_inputs).to(device)

    # Run generation for the entire batch
    with torch.no_grad():
        forecast_standardized = model.predict(batch_tensor, args.prediction_length)
        # forecast_standardized has shape (batch_size, num_quantiles, prediction_length)
        # for example: [ (batch size is 1)
        #                  [-1.21, -1.18, -1.17], # we have 3 values (3 horizons for every quantile)
        #                  [...], [...], [...]
        #               ]

        # De-standardize each forecast in the batch
        # We iterate along the batch dimension of forecast_standardized

    for i, forecast_tensor in enumerate(forecast_standardized):
        mean_val, std_val = means_stds[i]
        
        forecast_np = forecast_tensor.detach().cpu().numpy()  # shape: (num_quantiles, prediction_length), num_quantiles = 9
        forecast_np = forecast_np * std_val + mean_val


        
        std_forecast_np = forecast_np.std(axis=0) # shape: (prediction_length,)
        diff_q8_q0 = forecast_np[8] - forecast_np[0] # shape: (prediction_length,). We have the difference between the first and last quantile for H=1, H=2 and H=3
        diff_q6_q2 = forecast_np[6] - forecast_np[2]
        diff_q5_q3 = forecast_np[5] - forecast_np[3]

        # Select quantile index 4 (assumed median quantile)
        quantile_for_prediction = 4

        median = forecast_np[quantile_for_prediction] # Here we take the median (5th quantile) as the final result
        
        # Now handle the 'return_type'
        if args.return_type == "median":
            final_result = {"median": median, 
                            "std": std_forecast_np,
                            "diff_q8_q0": diff_q8_q0,
                            "diff_q6_q2": diff_q6_q2,
                            "diff_q5_q3": diff_q5_q3}
        else:
            raise ValueError(f"Unknown return_type: {args.return_type}")

        all_forecasts.append(final_result)

    return all_forecasts