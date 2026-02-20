## Dataset Setup

This project uses renewable energy time series data from the World Bank database.

### Step 1 – Download the Dataset
Download the CSV file from the World Bank Data Portal.

### Step 2 – Place the File
Move the CSV file into the project root directory.

Example structure:

renewable-energy-arima/
|
----> projem4.py
----> clean_fuels.csv
----> outputs/
----> README.md

### Step 3 – Update File Path (if needed)

Inside `projem4.py`, update the following variable:

DATA_FILE = "clean_fuels.csv"

Make sure the file path matches the location of your dataset.

## Project Overview

This project analyzes Turkey’s renewable energy time series data obtained from the World Bank database.

The workflow includes:

- Automatic year column detection
- Missing value handling (interpolation)
- Outlier treatment using IQR
- Train/Test split for time series
- ARIMA model selection (grid search with AIC)
- Forecasting for 2023–2030
- Confidence interval visualization (95%)
- Residual diagnostics (ACF, QQ plot, Ljung-Box test)
- Automatic policy recommendations based on forecast trends

The goal is to combine statistical time series modeling with policy-driven interpretation.
