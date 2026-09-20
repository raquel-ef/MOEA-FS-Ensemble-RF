import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from platypus import unique, nondominated

seed_value = 0


def split_features_target(df):
    """
    Splits a DataFrame into feature variables (X) and target variable (Y).

    Parameters:
        df (pd.DataFrame): The input DataFrame where the last column is assumed to be the target.

    Returns:
        X (pd.DataFrame): Features (all columns except the last).
        Y (pd.DataFrame): Target (the last column).
    """
    return df.iloc[:, :-1], df.iloc[:, -1:]


def calculateErrorsSteps(test_y, pred):
    """
    Calculates regression evaluation metrics between true and predicted target values.

    Parameters:
        test_y (pd.Series | pd.DataFrame | np.ndarray): Ground truth values.
        pred (pd.Series | pd.DataFrame | np.ndarray): Model predictions.

    Returns:
        mae (float): Mean Absolute Error.
        rmse (float): Root Mean Squared Error.
        cc (float): Pearson Correlation Coefficient.
    """
    mae = mean_absolute_error(test_y, pred)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    cc = np.corrcoef(test_y.T, pred.T)[1,0]    
    return mae, rmse, cc


def lags(df_original, n_lag):
    """
    Generates lag features for all variables in the DataFrame (including the target variable) and appends the unlagged original target column at the end.

    Parameters:
        df_original (pd.DataFrame): Input dataset where the last column is the target.
        n_lag (int): Number of lag steps to generate for every column.

    Returns:
        df_lags (pd.DataFrame): DataFrame containing generated lag columns (Lag_<col>_1, ..., Lag_<col>_n) for all columns, followed by the original unlagged target column.
    """
    lagged_cols = []

    # Generate lagged columns
    for col in df_original.columns:
        base_name = col.split(' ', 1)[0]
        for j in range(1, n_lag + 1):
            lagged_cols.append(df_original[col].shift(j).rename(f'Lag_{base_name}_{j}'))

    # Concatenate lagged columns
    df_lags = pd.concat(lagged_cols, axis=1)

    # Add target column at the end of the DataFrame
    last_col = df_original.columns[-1]
    df_lags[last_col] = df_original[last_col]

    return df_lags


def predictionsModels(results, Vx, Vy):
    """
    Trains multiple Random Forest regressor sub-models on feature subsets defined by non-dominated optimization solutions, 
    returning their predictions alongside the true target values.

    Parameters:
        results: Multi-objective optimization results (e.g., Pareto-optimal solutions) containing boolean feature selection masks.
        Vx (pd.DataFrame): Feature matrix.
        Vy (pd.Series | pd.DataFrame): Target variable values.

    Returns:
        predicciones (pd.DataFrame): DataFrame where each numbered column contains predictions from a Random Forest trained on a 
        specific Pareto-optimal feature subset, with an additional 'output' column containing the true target values (Vy).
    """

    predicciones = pd.DataFrame()
    nondominated_results = unique(nondominated(results))

    for res, idx in zip(nondominated_results, range(len(nondominated_results))):
        mask = [sol[0] for sol in res.variables]  # Create a mask from variables
        Vx_selected = Vx.loc[:, mask]  # Select features
        
        # Train the RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=seed_value)
        model.fit(Vx_selected, Vy)
        
        # Predict for all rows at once
        predictions = model.predict(Vx_selected)
        predicciones[idx] = predictions

    predicciones['output'] = Vy.reset_index(drop=True)
    return predicciones


def predictionsModelsX(results, Vx, Vy):
    """
    Generates meta-features (ensemble sub-model predictions) without attaching the true target column, specifically designed to 
    prepare inputs for a top-level meta-model in a stacking framework.
    
    Parameters:
        results: Multi-objective optimization results containing feature selection masks.
        Vx (pd.DataFrame): Feature matrix.
        Vy (pd.Series | pd.DataFrame): Target variable values used for fitting base models.
        
    Returns:
        predicciones (pd.DataFrame): DataFrame containing only the sub-model predictions (columns indexed 0, 1, ..., N-1), 
        acting as synthetic meta-features for ensemble modeling.
    """
    predicciones = pd.DataFrame()
    nondominated_results = unique(nondominated(results))

    for res, idx in zip(nondominated_results, range(len(nondominated_results))):
        mask = [sol[0] for sol in res.variables]  # Create a mask from variables
        Vx_selected = Vx.loc[:, mask]  # Select features 
        
        # Train the RandomForest model
        model = RandomForestRegressor(n_estimators=100, random_state=seed_value)
        model.fit(Vx_selected, Vy)
        
        # Predict for all rows at once
        predictions = model.predict(Vx_selected)
        predicciones[idx] = predictions

    return predicciones


def predictionshstepsaheadEnsemble(testX, testy, results, model, n_steps):
    """
    Executes iterative h-step ahead time-series forecasting using a two-stage stacked ensemble model, 
    combining feature-subset sub-models (predictionsModelsX) with a top-level meta-model.

    Parameters:
        testX (pd.DataFrame): Test set feature matrix (including lag variables).
        testy (pd.DataFrame): Ground truth target values for the test set.
        results: Non-dominated optimization solutions used to construct feature masks for base models.
        model (Estimator): Top-level meta-model trained to combine base model predictions into final forecasts.
        n_steps (int): The forecasting horizon (h-steps ahead).
    
    Returns:
        results_df (pd.DataFrame): Table listing performance metrics (RMSE, MAE, CC) for each step in the forecasting horizon.
        predictions (pd.DataFrame): Target predictions for each forecast horizon (pred1, pred2, ..., pred{n_steps+1}).
        test_X (pd.DataFrame): Test feature matrix updated with recursively injected lagged predictions.
    """
    # Extract lags from the column names
    predicted_attribute = "output"
    selected_lags = [
        int(col.split("_")[2])
        for col in testX.filter(regex=f"{predicted_attribute}.*").columns
    ]

    # Reset indices for testX and testy
    test_X = testX.reset_index(drop=True)
    test_y = testy.reset_index(drop=True)

    # Initialize results
    predictions = pd.DataFrame()
    results_df = pd.DataFrame(columns=["RMSE", "MAE", "CC"])

    if selected_lags:  # Proceed only if there are lagged features
        # 1-step ahead prediction
        ensemble_features = predictionsModelsX(results, test_X, test_y)
        predictions["pred1"] = model.predict(ensemble_features).ravel()

        mae, rmse, cc = calculateErrorsSteps(test_y, predictions[["pred1"]])
        results_df = pd.concat(
            [results_df, pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "CC": [cc]})],
            ignore_index=True,
        )

        # Multi-step ahead predictions
        for step in range(2, n_steps + 2):
            for lag in range(1, step):
                lag_col = f"Lag_{predicted_attribute}_{step - lag}"
                if step - lag in selected_lags:
                    test_X[lag_col] = predictions[f"pred{lag}"].shift(step - lag)

            valid_rows = test_X.dropna()
            ensemble_features = predictionsModelsX(results, valid_rows, test_y.iloc[(step-1):])
            step_predictions = model.predict(ensemble_features)

            # Insert NaN for shifted rows
            step_predictions_full = np.full(len(test_X), np.nan)
            step_predictions_full[(step - 1) :] = step_predictions
            predictions[f"pred{step}"] = step_predictions_full

            # Calculate errors
            valid_predictions = predictions.iloc[(step - 1) :]
            valid_test_y = test_y.iloc[(step - 1) :]
            mae, rmse, cc = calculateErrorsSteps(
                valid_test_y, valid_predictions[[f"pred{step}"]]
            )
            results_df = pd.concat(
                [results_df, pd.DataFrame({"RMSE": [rmse], "MAE": [mae], "CC": [cc]})],
                ignore_index=True,
            )

    return results_df, predictions, test_X
