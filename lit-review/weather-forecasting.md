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
### Spatial Resolution:
- `0.25° x 0.25°` latitude/longitude (depending on input data)
- Output likely to have a resolution of 3 km
### Spatial Domain:
- Regional forecasts have a certain bounding box
- Can arbitrarily choose a 60 km x 60 km region including KW region (TO-DO: look into what other papers have done, tradeoffs, etc.)

## Data Sources & Provenance
### Datasets
- HRRR (model input, forecasts include 0-48 hour forecasts -> goes into a model which outputs a corrected forecast)
- Training the model involves ground truth (what our loss function compares)
- Eric D. Soulis station observations (same as benchmarking)
- ERA5 weather observations (i.e, ground truth)
- (TO-DO: add guide for loading-in ERA5 & HRRR data for training)
- (TO-DO: look into how to incorporate ERA5 & Eric D. Soulis for ground truth)
- (TO-DO: look into what operational details we want to flesh out & how that can inform data requirements, etc.)


### Storage & Formats
### Input Features
- 10u (NS) 10v (EW) 
- May contain more features than what we target, may also depend on different teams
### Temporal Split

