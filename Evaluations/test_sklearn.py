from sktime.forecasting.ttm import TinyTimeMixerForecaster
from sktime.datasets import load_tecator
# load multi-index dataset
y = load_tecator(
    return_type="pd-multiindex",
    return_X_y=False
)
y.drop(['class_val'], axis=1, inplace=True)
# global forecasting on multi-index dataset
forecaster = TinyTimeMixerForecaster(
    model_path=None,
    fit_strategy="full",
    config={
            "context_length": 8,
            "prediction_length": 2
    },
    training_args={
        "num_train_epochs": 1,
        "output_dir": "test_output",
        "per_device_train_batch_size": 32,
    },
) 
# model initialized with random weights due to None model_path
# and trained with the full strategy.
forecaster.fit(y, fh=[1, 2, 3]) 
y_pred = forecaster.predict() 