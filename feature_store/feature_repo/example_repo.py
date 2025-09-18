# Это пример файла с определением признаков (feature definition)

import os

from datetime import timedelta

import numpy as np
import pandas as pd

from feast import (
    Entity,
    FeatureService,
    FeatureView,
    Field,
    FileSource,
    PushSource,
    RequestSource,
)
from feast.feature_logging import LoggingConfig
from feast.infra.offline_stores.file_source import FileLoggingDestination
from feast.on_demand_feature_view import on_demand_feature_view
from feast.types import Float32, Float64, Int64

REPO_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(REPO_PATH, "data")

# Определяем сущность для водителя. Сущность можно рассматривать как первичный ключ,
# который используется для получения признаков
driver = Entity(name="driver", join_keys=["driver_id"])

# Читаем данные из parquet файлов. Parquet удобен для локальной разработки.
# Для промышленного использования можно использовать любое хранилище данных,
# например BigQuery. Подробнее смотрите в документации Feast
driver_stats_source = FileSource(
    name="driver_hourly_stats_source",
    path=os.path.join(DATA_PATH, "driver_stats.parquet"),
    timestamp_field="event_timestamp",
    created_timestamp_column="created",
)

# Наши parquet файлы содержат примеры данных, включающие столбец driver_id,
# временные метки и три столбца с признаками. Здесь мы определяем Feature View,
# который позволит нам передавать эти данные в нашу модель онлайн
# ------------------------------
# 1. Feature View: метрики качества водителя
# ------------------------------
driver_quality_fv = FeatureView(
    name="driver_quality_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32, description="Коэффициент конверсии"),
        Field(name="acc_rate", dtype=Float32, description="Коэффициент точности"),
    ],
    online=True,
    source=driver_stats_source,
    tags={"group": "driver_quality"},
)

# ------------------------------
# 2. Feature View: активность водителя
# ------------------------------
driver_activity_fv = FeatureView(
    name="driver_activity_stats",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_daily_trips", dtype=Int64, description="Среднее число поездок в день"),
    ],
    online=True,
    source=driver_stats_source,
    tags={"group": "driver_activity"},
)

# Определяем представление признаков по требованию, которое может генерировать
# новые признаки на основе существующих представлений и признаков из RequestSource
@on_demand_feature_view(
    sources=[driver_quality_fv, driver_activity_fv],
    schema=[
        Field(name="efficiency_index", dtype=Float64),
        Field(name="risk_score", dtype=Float64),
    ],
)
def driver_efficiency_metrics(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["efficiency_index"] = inputs["conv_rate"] + inputs["avg_daily_trips"]
    df["risk_score"] = 1 - inputs["acc_rate"] * np.log(1+inputs["avg_daily_trips"])
    return df


# FeatureService группирует признаки в версию модели
driver_activity_v1 = FeatureService(
    name="driver_activity_v1",
    features=[
        driver_quality_fv[["conv_rate"]],  # Выбирает подмножество признаков из представления
        driver_efficiency_metrics,  # Выбирает все признаки из представления
    ],
    logging_config=LoggingConfig(
        destination=FileLoggingDestination(path=DATA_PATH)
    ),
)
driver_activity_v2 = FeatureService(
    name="driver_activity_v2", features=[driver_quality_fv, driver_activity_fv, driver_efficiency_metrics]
)

# Определяет способ отправки данных (доступных офлайн, онлайн или обоих типов) в Feast
driver_stats_push_source = PushSource(
    name="driver_stats_push_source",
    batch_source=driver_stats_source,
)

# Определяет слегка измененную версию представления признаков, описанного выше,
# где источник был изменен на push source. Это позволяет напрямую отправлять
# свежие признаки в онлайн-хранилище для этого представления признаков
driver_quality_fresh_fv = FeatureView(
    name="driver_quality_stats_fresh",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="conv_rate", dtype=Float32, description="Коэффициент конверсии"),
        Field(name="acc_rate", dtype=Float32, description="Коэффициент точности"),
    ],
    online=True,
    source=driver_stats_push_source,
    tags={"group": "driver_quality"},
)
# ------------------------------
# 2. Feature View: активность водителя
# ------------------------------
driver_activity_fresh_fv = FeatureView(
    name="driver_activity_stats_fresh",
    entities=[driver],
    ttl=timedelta(days=1),
    schema=[
        Field(name="avg_daily_trips", dtype=Int64, description="Среднее число поездок в день"),
    ],
    online=True,
    source=driver_stats_push_source,
    tags={"group": "driver_activity"},
)
# Определяем представление признаков по требованию, которое может генерировать
# новые признаки на основе существующих представлений и признаков из RequestSource
@on_demand_feature_view(
    sources=[driver_quality_fresh_fv, driver_activity_fresh_fv],
    schema=[
        Field(name="efficiency_index", dtype=Float64),
        Field(name="risk_score", dtype=Float64),
    ],
)
def driver_efficiency_metrics_fresh(inputs: pd.DataFrame) -> pd.DataFrame:
    df = pd.DataFrame()
    df["efficiency_index"] = inputs["conv_rate"] + inputs["avg_daily_trips"]
    df["risk_score"] = 1 - inputs["acc_rate"] * np.log(1+inputs["avg_daily_trips"])
    return df
# FeatureService группирует признаки в версию модели
driver_activity_v3 = FeatureService(
    name="driver_activity_v3", features=[driver_quality_fresh_fv, driver_activity_fresh_fv, driver_efficiency_metrics_fresh]
)

