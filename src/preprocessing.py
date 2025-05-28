import pandas as pd
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)
RANDOM_STATE = 42


def load_train_data():

    logger.info("Loading training data...")

    # Определяем тип признаков
    cat_columns = [
        "merch",
        "cat_id",
        "name_1",
        "name_2",
        "gender",
        "street",
        "one_city",
        "us_state",
        "post_code",
        "jobs",
    ]
    continuous_cols = [
        "amount",
        "lat",
        "lon",
        "population_city",
        "merchant_lat",
        "merchant_lon",
    ]

    # Импортируем датасет train
    train = pd.read_csv("./train_data/train.csv").drop(columns=["transaction_time"])

    logger.info("Raw train data imported. Shape: %s", train.shape)

    # Обрабатываем категориальные переменные
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    train[cat_columns] = encoder.fit_transform(train[cat_columns])

    # обработка числовых переменных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(train[continuous_cols])
    X_train_scaled_df = pd.DataFrame(
        X_train_scaled, columns=continuous_cols, index=train.index
    )
    train[continuous_cols] = X_train_scaled_df

    logger.info("Train data processed. Shape: %s", train.shape)

    return train, encoder, scaler


# главная функция предобработки
def run_preproc(encoder, scaler, input_df):

    # Определяем тип признаков
    cat_columns = [
        "merch",
        "cat_id",
        "name_1",
        "name_2",
        "gender",
        "street",
        "one_city",
        "us_state",
        "post_code",
        "jobs",
    ]
    continuous_cols = [
        "amount",
        "lat",
        "lon",
        "population_city",
        "merchant_lat",
        "merchant_lon",
    ]

    input_df = input_df.drop(columns="transaction_time")

    # Обрабатываем категориальные переменные
    input_df[cat_columns] = encoder.transform(input_df[cat_columns])
    logger.info("Categorical mean encoding completed. Output shape: %s", input_df.shape)

    # обработка числовых переменных
    X_test_scaled = scaler.transform(input_df[continuous_cols])
    X_test_scaled_df = pd.DataFrame(
        X_test_scaled, columns=continuous_cols, index=input_df.index
    )
    input_df[continuous_cols] = X_test_scaled_df
    output_df = input_df
    logger.info(
        "Continuous features preprocessing completed. Output shape: %s", output_df.shape
    )

    return output_df
