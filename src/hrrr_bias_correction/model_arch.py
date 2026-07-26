import importlib

N_LEADS = 49
CROP = 30
N_CHANNELS = 3  # HRRR input: t2m, u10, v10
N_TARGETS = 2  # bias outputs: t2m (°C), wind_speed (km/h)

input_shape = (None, N_LEADS, CROP, CROP, N_CHANNELS)  # (batch, lead, y, x, channel)


def build_model(input_shape=input_shape, n_outputs: int = N_TARGETS):
    """CNN-LSTM that maps HRRR crops → per-lead station bias.

    Output shape: (batch, 49, 2) — t2m bias and wind_speed bias.
    """
    if len(input_shape) == 5:
        sample_shape = input_shape[1:]
    elif len(input_shape) == 4:
        sample_shape = input_shape
    else:
        raise ValueError(
            "input_shape must be (lead, row, col, channel) or "
            "(batch, lead, row, col, channel)."
        )

    try:
        tf = importlib.import_module("tensorflow")
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "TensorFlow is required to build the model. Install dependencies in a "
            "Python 3.10-3.12 environment (3.11 recommended)."
        ) from exc

    model = tf.keras.Sequential()
    model.add(tf.keras.Input(shape=sample_shape))

    model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")
        )
    )
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")))

    model.add(
        tf.keras.layers.TimeDistributed(
            tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")
        )
    )
    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.MaxPooling2D(pool_size=(2, 2), padding="same")))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten()))

    model.add(tf.keras.layers.LSTM(64, return_sequences=True))

    model.add(tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=n_outputs, activation="linear")))

    return model
