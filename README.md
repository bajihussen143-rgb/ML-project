Titanic Data Preparation Pipeline — Data Studio

A portfolio-oriented, interactive dark navy/purple dashboard for the Titanic preprocessing project. It shows data previews, quality checks, transformation stages, split statistics, and downloadable CSV outputs.

Scope: data preparation only. This project does not train a survival prediction model.

Features





Premium dark navy / purple Streamlit interface



Overview metrics and survival-class visualization



Data Explorer for original, cleaned, imputed, encoded, and scaled data



Data quality report and missing-value comparison



Preprocessing Lab with train/test distribution and scaling statistics



CSV export center



Uses the existing titanic_pipeline.py functions



Run on Windows (VS Code)





Extract the ZIP and open the titanic_pipeline folder in VS Code.



Open Terminal → New Terminal.



Create and activate a virtual environment:

 python -m venv .venv
 .\.venv\Scripts\Activate.ps1

 If PowerShell blocks activation, use Command Prompt instead: .venv\Scripts\activate.bat.



Install dependencies:

 python -m pip install -r requirements.txt



Launch the dashboard:

 python -m streamlit run app.py



Open the local URL printed in the terminal (usually http://localhost:8501).

The first run may need an internet connection because Seaborn loads its sample Titanic dataset.

Pages





Overview — summary metrics, dataset preview, survival distribution, pipeline map.



Data Explorer — filter rows and inspect transformation stages.



Data Quality — missing values, duplicates, data types, and column audit.



Preprocessing Lab — transformation explanations and train/test statistics.



Exports — download encoded, split, and scaled CSV files.



About the Project — scope, technologies, and limitations.



Technical note about data leakage

The assignment-style pipeline imputes and encodes before splitting, while the scaler is fitted only on training features. For a stricter production workflow, fit imputers and encoders on training data only and transform the test set.

Portfolio checklist





Run the app and inspect every page.



Save a screenshot of the Overview page to screenshots/dashboard.png.



Run the automated tests.



Add the screenshot and a short project summary to GitHub.



Explain clearly that this project prepares data and does not train a prediction model.
