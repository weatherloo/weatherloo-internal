from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, TimeDistributed, Conv2D, MaxPooling2D, Flatten, LSTM, Dense

N_LEADS = 49
CROP = 30
N_CHANNELS = 3  # HRRR input: t2m, u10, v10
N_TARGETS = 2  # bias outputs: t2m (°C), wind_speed (km/h)

input_shape = (None, N_LEADS, CROP, CROP, N_CHANNELS)  # (batch, lead, y, x, channel)


def build_model(input_shape=input_shape, n_outputs: int = N_TARGETS):
    """CNN-LSTM that maps HRRR crops → per-lead station bias.

    Output shape: (batch, 49, 2) — t2m bias and wind_speed bias.
    """
    model = Sequential()
    model.add(Input(shape=input_shape[1:]))

    model.add(TimeDistributed(Conv2D(filters=16, kernel_size=(3, 3), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding="same")))

    model.add(TimeDistributed(Conv2D(filters=32, kernel_size=(3, 3), activation="relu", padding="same")))
    model.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), padding="same")))

    model.add(TimeDistributed(Flatten()))

    model.add(LSTM(64, return_sequences=True))

    model.add(TimeDistributed(Dense(units=n_outputs, activation="linear")))

    return model
