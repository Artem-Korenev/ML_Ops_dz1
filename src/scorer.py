import pandas as pd
import logging
import joblib

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger.info("Importing pretrained model...")
components = joblib.load("./models/model_classific.pkl")
model = components["model"]

# Определяем порог
model_th = 0.24
logger.info("Pretrained model imported successfully...")


# Функция предсказания
def make_pred(dt, path_to_file):

    # формируем датафрейм
    submission = pd.DataFrame(
        {
            "index": pd.read_csv(path_to_file).index,
            "prediction": (model.predict_proba(dt)[:, 1] > model_th) * 1,
        }
    )
    logger.info("Prediction complete for file: %s", path_to_file)

    return submission
