def test_core_modules_import():
    from models.ann_model import AdvancedANN
    from models.qinn_model import QINN
    from preprocessing.data_processor import DataProcessor
    from utils.config import CONFIG

    assert AdvancedANN is not None
    assert QINN is not None
    assert DataProcessor is not None
    assert CONFIG.PREDICTION_HORIZONS


def test_technical_indicators_do_not_backfill_rolling_values():
    import logging

    import pandas as pd

    from preprocessing.data_processor import DataProcessor
    from utils.config import CONFIG

    dates = pd.date_range("2024-01-01", periods=8, freq="D")
    df = pd.DataFrame(
        {
            "Open": range(100, 108),
            "High": range(101, 109),
            "Low": range(99, 107),
            "Close": range(100, 108),
            "Volume": [1000] * 8,
        },
        index=dates,
    )

    processor = DataProcessor(CONFIG, logging.getLogger("test"))
    out = processor.add_technical_indicators(df, windows=[5])

    assert out.loc[dates[0], "ma_5"] == 0
    assert out.loc[dates[4], "ma_5"] == 102
