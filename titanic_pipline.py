"""
Titanic Survivor Data Preparation Pipeline
==========================================
Prepares a Titanic dataset for downstream machine-learning models.
This script intentionally does NOT train a prediction model.

Required functions:
  load_data, validate_data, clean_text, drop_useless, handle_missing,
  encode_categoricals, split_data, scale_features, main
"""

from __future__ import annotations

import sys
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


TARGET = "survived"
REQUIRED_COLUMNS = {
    "survived", "pclass", "sex", "age", "sibsp", "parch",
    "fare", "embarked"
}


class DataValidationError(Exception):
    """Raised when the Titanic dataset does not meet the expected schema."""


def load_data() -> pd.DataFrame:
    """Load the Titanic dataset provided by Seaborn."""
    try:
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "Seaborn is not installed. Run: pip install -r requirements.txt"
        ) from exc

    try:
        data = sns.load_dataset("titanic")
    except Exception as exc:
        raise RuntimeError(
            "Could not load Seaborn's Titanic dataset. Check your internet "
            "connection and try again. Seaborn may download the dataset "
            "the first time it is used."
        ) from exc

    return data.copy()


def validate_data(data: pd.DataFrame) -> None:
    """Check that the input is a DataFrame and required columns exist."""
    if not isinstance(data, pd.DataFrame):
        raise DataValidationError("Input data must be a pandas DataFrame.")

    missing = REQUIRED_COLUMNS.difference(data.columns)
    if missing:
        raise DataValidationError(
            "Required column(s) missing: " + ", ".join(sorted(missing))
        )

    if data.empty:
        raise DataValidationError("The dataset contains no rows.")

    if data[TARGET].isna().any():
        raise DataValidationError(
            "The target column 'survived' contains missing values."
        )


def clean_text(data: pd.DataFrame) -> pd.DataFrame:
    """Normalize text values: trim whitespace and use lowercase."""
    result = data.copy()

    text_columns = result.select_dtypes(
        include=["object", "string", "category"]
    ).columns
    for column in text_columns:
        # Convert category columns to object first so normalization works
        # consistently for Seaborn's categorical columns and ordinary strings.
        result[column] = result[column].astype("object").map(
            lambda value: value.strip().lower()
            if isinstance(value, str)
            else value
        )

    return result


def drop_useless(data: pd.DataFrame) -> pd.DataFrame:
    """
    Remove target leakage, duplicate/redundant fields, and mostly-empty fields.

    'alive' duplicates the target 'survived'; 'deck' is mostly missing.
    'class' duplicates 'pclass'; 'embark_town' duplicates 'embarked';
    'adult_male' and 'who' overlap with information in age/sex.
    """
    result = data.copy()

    leakage_or_redundant = [
        "alive", "deck", "class", "embark_town", "adult_male", "who"
    ]
    columns_to_drop = [c for c in leakage_or_redundant if c in result.columns]

    # General safeguard: drop any non-target column that is more than 50%
    # missing. This follows the brief's "mostly empty" requirement.
    mostly_empty = [
        column
        for column in result.columns
        if column != TARGET and result[column].isna().mean() > 0.50
    ]
    columns_to_drop = sorted(set(columns_to_drop + mostly_empty))

    return result.drop(columns=columns_to_drop)


def handle_missing(data: pd.DataFrame) -> pd.DataFrame:
    """Fill numeric missing values with median and categorical with mode."""
    result = data.copy()

    for column in result.columns:
        if column == TARGET:
            # The target should not be imputed; it was checked in validate_data.
            continue

        if pd.api.types.is_numeric_dtype(result[column]):
            if result[column].isna().any():
                median = result[column].median()
                if pd.isna(median):
                    raise DataValidationError(
                        f"Numeric column '{column}' contains no usable values."
                    )
                result[column] = result[column].fillna(median)
        else:
            if result[column].isna().any():
                modes = result[column].mode(dropna=True)
                if modes.empty:
                    raise DataValidationError(
                        f"Categorical column '{column}' contains no usable values."
                    )
                result[column] = result[column].fillna(modes.iloc[0])

    return result


def encode_categoricals(data: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical columns:
      - binary map: sex (female=0, male=1)
      - ordinal: pclass is already ordinal (1st, 2nd, 3rd class)
      - one-hot: remaining categorical features, e.g. embarked
    """
    result = data.copy()

    if "sex" in result.columns:
        sex_values = set(result["sex"].dropna().unique())
        if not sex_values.issubset({"female", "male"}):
            raise DataValidationError(
                "Unexpected value in 'sex'; expected 'female' or 'male'."
            )
        result["sex"] = result["sex"].map({"female": 0, "male": 1})

    # 'pclass' is already ordinal in the source dataset: 1, 2, 3.
    # Convert bools to integers if present.
    for column in result.select_dtypes(include=["bool"]).columns:
        result[column] = result[column].astype(int)

    categorical_columns = result.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()
    categorical_columns = [c for c in categorical_columns if c != TARGET]

    if categorical_columns:
        result = pd.get_dummies(
            result,
            columns=categorical_columns,
            drop_first=False,
            dtype=int
        )

    # Keep the target as an integer 0/1 column, never one-hot encode it.
    result[TARGET] = result[TARGET].astype(int)

    return result


def split_data(
    data: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a stratified 80/20 train-test split."""
    if TARGET not in data.columns:
        raise DataValidationError(f"Target column '{TARGET}' is missing.")

    X = data.drop(columns=[TARGET])
    y = data[TARGET]

    if X.empty or X.shape[1] == 0:
        raise DataValidationError("No feature columns remain after preprocessing.")

    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
    except ValueError as exc:
        raise DataValidationError(
            "Could not create a stratified split. Check that each target class "
            "has enough rows and that test_size is valid."
        ) from exc

    return X_train, X_test, y_train, y_test


def scale_features(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, StandardScaler]:
    """Fit StandardScaler on training features only, then transform both sets."""
    scaler = StandardScaler()

    train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=X_train.columns,
        index=X_train.index
    )
    test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=X_test.columns,
        index=X_test.index
    )

    return train_scaled, test_scaled, scaler


def main() -> None:
    """Run all preprocessing stages and print a concise summary."""
    try:
        data = load_data()
        print(f"Loaded dataset: {data.shape[0]} rows, {data.shape[1]} columns")

        validate_data(data)
        data = clean_text(data)
        data = drop_useless(data)
        data = handle_missing(data)
        data = encode_categoricals(data)

        X_train, X_test, y_train, y_test = split_data(data)
        X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

        print("\nPipeline completed successfully.")
        print(f"Training features: {X_train_scaled.shape}")
        print(f"Testing features:  {X_test_scaled.shape}")
        print(f"Training target:   {y_train.shape}")
        print(f"Testing target:    {y_test.shape}")
        print(f"Encoded feature columns: {list(X_train_scaled.columns)}")
        print("\nMissing values after preprocessing:")
        print(f"  Train: {int(X_train_scaled.isna().sum().sum())}")
        print(f"  Test:  {int(X_test_scaled.isna().sum().sum())}")
        print("\nTarget proportions (train):")
        print(y_train.value_counts(normalize=True).sort_index().round(3))
        print("\nTarget proportions (test):")
        print(y_test.value_counts(normalize=True).sort_index().round(3))
        print("\nNo prediction model was trained; the prepared data is ready for downstream use.")

    except (DataValidationError, RuntimeError) as exc:
        print(f"Pipeline error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
