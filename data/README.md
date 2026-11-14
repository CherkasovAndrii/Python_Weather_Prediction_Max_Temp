# Weather prediction dataset

`weather_prediction_dataset.csv` contains daily weather observations from **18 European locations**,
**2000-01-01 – 2010-01-01** (3,654 days, 165 columns, no `NaN`s).

Source: Huber, F. et al. (2022). *Weather prediction dataset* (v5). Zenodo.
<https://doi.org/10.5281/zenodo.7525955>. License: [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
Original data: European Climate Assessment & Dataset (ECA&D), Klein Tank et al. (2002), <https://www.ecad.eu>.

## Columns

`DATE` (`YYYYMMDD`) and `MONTH`, followed by `<LOCATION>_<variable>` columns. Not every location has every variable.

| Variable | Unit in this CSV |
|----------|------------------|
| `temp_mean`, `temp_min`, `temp_max` | °C |
| `cloud_cover` | oktas (0–8, 9 = sky obscured) |
| `humidity` | fraction (0–1) |
| `pressure` | 1000 hPa (1.013 = 1013 hPa) |
| `global_radiation` | 100 W/m² |
| `precipitation` | 10 mm |
| `sunshine` | hours |
| `wind_speed`, `wind_gust` | m/s |

Locations: Basel, Budapest, De Bilt, Dresden, Düsseldorf, Heathrow, Kassel, Ljubljana, Maastricht,
Malmö, Montélimar, München, Oslo, Perpignan, Roma, Sonnblick, Stockholm, Tours.

## Known data issues

A few missing observations are encoded as physically impossible values. `maxtemp.data.clean_dataset`
replaces them with the last valid observation:

| Column | Invalid values | Example |
|--------|---------------:|---------|
| `STOCKHOLM_sunshine` | 29 | −1.7 hours |
| `STOCKHOLM_pressure` | 3 | −0.099 |
| `STOCKHOLM_cloud_cover` | 2 | −99 oktas (in the test period) |
| `TOURS_pressure` | 1 | 0.0003 (in the test period) |
