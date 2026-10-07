import pandas as pd

DATA_PATH = "data/cleaned_supply_chain_data.csv"


def prepare_features():

    df = pd.read_csv(DATA_PATH)

    # Target variable
    y = df["delay"]

    # Remove ID and target columns
    # delay_days is removed to avoid target leakage
    X = df.drop(
        columns=["shipment_id", "delay", "delay_days"]
    )

    categorical_columns = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    print("Feature engineering completed!")
    print("Dataset shape:", X.shape)

    print("\nNumerical features:")
    print(numerical_columns)

    print("\nCategorical features:")
    print(categorical_columns)

    print("\nTarget shape:", y.shape)

    return X, y, categorical_columns, numerical_columns


if __name__ == "__main__":
    prepare_features()