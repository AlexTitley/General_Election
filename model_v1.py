import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error


# Preparing the data into X feature matrix and y target vectors

df = pd.read_csv("train_test_dataset.csv")

# Filling in missung values
df = df.fillna(0)

# Turning the string data to numerical values for the model to use
encoded_region = LabelEncoder()
df['region_encoded'] = encoded_region.fit_transform(df['region'])

encoded_prev_winner = LabelEncoder()
df['prev_winner_encoded'] = encoded_prev_winner.fit_transform(df['prev_winner'])

# (y) - the target columns
target_cols = ['con_share', 'lab_share', 'ld_share', 'snp_pc_share', 'reform_share', 
    'green_share', 'other_share']
y = df[target_cols]

# Features to be used in the model - make sure to match it with both CSV files
feature_cols = ['year', 'boundary_set', 'region_encoded', 'prev_winner_encoded',
    'prev_con_share', 'prev_lab_share', 'prev_ld_share', 'prev_snp_pc_share',
    'prev_reform_share', 'prev_green_share', 'prev_other_share', 'prev_margin',
    'prev_rel_con', 'prev_rel_lab', 'prev_rel_ld', 'prev_rel_snp_pc',
    'prev_rel_reform', 'prev_rel_green', 'prev_rel_other',
    'nat_swing_con', 'nat_swing_lab', 'nat_swing_ld', 'nat_swing_snp_pc',
    'nat_swing_reform', 'nat_swing_green', 'nat_swing_other',
    'house_price_rel', 'income_rel', 'employment_rel',
    'pop_0_15_rel', 'pop_16_64_rel', 'pop_65_plus_rel']

X = df[feature_cols]

# Training and testing the model

# Data split (80% for training, 20% for testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
model.fit(X_train, y_train)

# Evaluate on test set
raw_test_preds = model.predict(X_test)
test_preds_df = pd.DataFrame(raw_test_preds, columns=target_cols, index=X_test.index)

# If reform/green party stood in the constituency then the share is multiplied by 1, if not stood then share is multiplied by 0
if 'reform_stood' in X_test.columns:
    test_preds_df['reform_share'] = test_preds_df['reform_share'] * X_test['reform_stood']
if 'green_stood' in X_test.columns:
    test_preds_df['green_share'] = test_preds_df['green_share'] * X_test['green_stood']

# Normalising test predictions to sum to 1.0 (100%)
test_preds_normalized = test_preds_df.div(test_preds_df.sum(axis=1), axis=0)

# Calculating Mean Absolute Error (MAE) to use in monte carlo simulation
mae = mean_absolute_error(y_test, test_preds_normalized)
print(f"Mean Absolute Error: {mae:.4f} (percentage points)")

# Preparing data for the prediction year

future_df = pd.read_csv("prediction_dataset.csv")

future_df['region_encoded'] = encoded_region.transform(future_df['region'])
future_df['prev_winner_encoded'] = encoded_prev_winner.transform(future_df['prev_winner'])

X_future = future_df[feature_cols]
X_future = X_future.fillna(0)


# Prediction for next general election

future_raw_preds = model.predict(X_future)
future_preds_df = pd.DataFrame(future_raw_preds, columns=target_cols, index=future_df.index)

# Normalising future predictions to sum to 1.0 (100%)
future_preds_normalized = future_preds_df.div(future_preds_df.sum(axis=1), axis=0)

# ==========================================
# MONTE CARLO SIMULATION FOR PROBABILITIES
# ==========================================
n_simulations = 1000
error_margin = mae  

# FIX: Use 0.0 instead of 0 to ensure the dataframe accepts decimal probabilities
win_probabilities = pd.DataFrame(0.0, index=future_df.index, columns=target_cols)

for idx in future_preds_normalized.index:
    base_prediction = future_preds_normalized.loc[idx].values
    
    # Generate 1,000 sets of random noise
    noise = np.random.normal(loc=0, scale=error_margin, size=(n_simulations, len(target_cols)))
    simulated_shares = base_prediction + noise
    simulated_shares = np.clip(simulated_shares, 0, None)
    
    # Find winners of simulations
    simulated_winners = np.argmax(simulated_shares, axis=1)
    winner_counts = np.bincount(simulated_winners, minlength=len(target_cols))
    
    # Convert to probabilities
    win_probabilities.loc[idx] = winner_counts / n_simulations

# Clean up column names
win_probabilities.columns = [f"prob_win_{col.replace('_share', '')}" for col in target_cols]

# ==========================================
# ASSEMBLE FINAL DELIVERABLE
# ==========================================
party_map = {
    'con_share': 'Conservative', 'lab_share': 'Labour', 
    'ld_share': 'Liberal Democrat', 'snp_pc_share': 'SNP/PC',
    'reform_share': 'Reform', 'green_share': 'Green', 'other_share': 'Other'
}

final_forecast = future_df[['constituency_id', 'name']].copy()

# Add predicted base vote shares
for col in target_cols:
    final_forecast[f"pred_{col}"] = future_preds_normalized[col].values

# Add probabilities
final_forecast = pd.concat([final_forecast, win_probabilities], axis=1)

# Find most likely winner
final_forecast['forecasted_winner'] = win_probabilities.idxmax(axis=1).apply(
    lambda x: party_map[x.replace('prob_win_', '') + '_share']
)

print("\nFinal Forecast Preview:")
print(final_forecast[['name', 'forecasted_winner', 'prob_win_con', 'prob_win_lab', 'prob_win_ld']].head())

final_forecast.to_csv("election_forecast_with_probabilities.csv", index=False)
print("\nSuccess! Saved to 'election_forecast_with_probabilities.csv'.")