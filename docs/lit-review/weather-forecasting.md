## Target Framework
### Target Variables: 
- `2m temperature` (°C), higher priority
- `10 m wind speed` (km/h)
- `Accumulated Precipitation` (mm)
### Forecast Horizon:
- `t + 48` hours (assuming HRRR is used for training & inference)
### Temporal Resolution:
- `6-hour` blocks
- Start with hourly forecasts --> 15-min forecasts eventually
- Every 6 hours, model is re-initialized: 1 prediction is created for each of the 48 hours
### Spatial Resolution:
- `0.25° x 0.25°` latitude/longitude (depending on input data)
- Output likely to have a resolution of 3 km
### Spatial Domain:
- Can arbitrarily choose a 60 km x 60 km region including KW region (TO-DO: look into what other papers have done, tradeoffs, etc.)
- Regional forecasts have a certain bounding box, it is recommendable to choose a 300 x 300 km bounding box for short-term forecasting (2-48 hrs). 
- During the training process, experiment with boxes to determine an optimal size.
- Bounding box: Top-Left (NW): 44.8159° N, -82.3481° W Top-Right (NE): 44.8159° N, -78.6853° W Bottom-Right (SE): 42.1175° N, -78.6853° W Bottom-Left (SW): 42.1175° N, -82.3481° W
### Challenges to address:
- Given our forecast horizon is t + 48 hours (up to 2 days in the future)
- The accuracy when nowcasting is more dependent on current atmospheric 
- Accuracy when forecasting from 6-48 hours is more dependent on correcting HRRR output
- Back this with what papers have done & look into what ECMWF AIFS does
- HyperDS is trained by incorporating training & weather station observations, look into how we can incorporate an approach to get output as a continuous scale modelling of the meteorological grid

## Data Sources & Provenance
### Datasets
- HRRR (model input, forecasts include 0-48 hour forecasts -> goes into a model which outputs a corrected forecast)
- Training the model involves ground truth (what our loss function compares)
- Eric D. Soulis station observations (same as benchmarking)
- ERA5 weather observations (i.e, ground truth)
- (TO-DO: add guide for loading-in ERA5 & HRRR data for training)
- (TO-DO: look into how to incorporate ERA5 & Eric D. Soulis for ground truth)
- (TO-DO: look into what operational details we want to flesh out & how that can inform data requirements, etc.)
- Operational details: potential to correct HRRR predictions over a longer time-span (i.e: 2-48 hours) + nowcasting head for corrections of forecasts within 0-6 hrs of current time


### Storage & Formats
### Input Features
- 10u (NS) 10v (EW)
- May contain more features than what we target, may also depend on different teams
### Temporal Split
- *Vaguely*: train on 2025, validate on 2026 (?)

