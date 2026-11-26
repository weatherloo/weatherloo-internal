# Weight-space programming pilot (November 2026)

Question: can a small direct edit to an existing forecaster's weights repair a
Waterloo-area error and improve forecasts it has not seen? These are exploratory
tests, not evidence that an LLM understands the whole model's weight space.

## What was tested

1. **Open pretrained model.** IBM's [TinyTimeMixer R1](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r1)
   (805,280 parameters) forecasts temperature from 512 past hourly readings.
   At daily 00Z origins, we scored +6h, +12h, and +24h. The agent inspected
   Soulis 2024 errors, then froze two edits to the actual checkpoint: three
   output-head bias values, or a gain and bias change to three output-head
   rows. The gain values came from a least-squares diagnostic on 2024; the
   resulting tensor changes were made directly, without model training.
   A constant output correction is a control. The edited checkpoint was then
   evaluated on 2025 Soulis and CYYZ without further tuning.
2. **Small weather-specific model.** A 65-parameter neural network was trained
   on January–May 2025 GFS temperature forecasts and Soulis observations to
   predict GFS bias. The agent inspected June–August errors and froze a
   one-bias edit and a two-weight edit involving one hidden unit. The edit
   magnitudes were fitted to development residuals without backpropagation.
   September–December was held out. The unchanged Soulis model and edits were
   also tested at CYYZ. This tests a different route: sparse repair of a
   model already fitted to local weather data.

In experiment 2, `raw GFS = observation + published bias` reconstructs the
forecast from the repository's published per-initialization benchmark pairs.
Only the forecast and issuance-time features enter the correction model. A
four-day gap separates train/development/test by forecast initialization,
longer than the 72-hour forecast horizon.

## Held-out results

MAE in °C; lower is better. TinyTimeMixer scores average the three horizons.
Soulis hourly data only support 148 eligible origins in 2025 (January–August);
CYYZ supports 364 across the full year. The edit was selected on Soulis 2024.

| TinyTimeMixer, 2025 | Original | Bias weight edit | Gain weight edit | Plain output bias |
|---|---:|---:|---:|---:|
| Soulis, 148 origins | 3.104 | 3.070 | **3.055** | 3.072 |
| CYYZ, 364 origins | 2.461 | 2.473 | **2.443** | 2.481 |

For the gain edit, weekly-block bootstrap intervals for MAE improvement over
the original include zero: Soulis +0.0495 °C [−0.0060, +0.1093], CYYZ +0.0180 °C
[−0.0228, +0.0585]. The bias edit also overlaps the plain output correction.

The GFS model scores all twelve 6–72h leads, with edits selected on Soulis
June–August and evaluated on September–December:

| GFS correction, 2025 test | Raw GFS | 65-param model | One-bias edit | Two-weight edit | Half-strength two-weight edit |
|---|---:|---:|---:|---:|---:|
| Soulis, 4,399 pairs | 1.874 | 1.816 | 1.791 | 2.015 | **1.766** |
| CYYZ, 5,634 pairs | **1.099** | 1.354 | 1.344 | 1.946 | 1.501 |

The Soulis one-bias edit's gain over the unchanged tiny model was +0.0250 °C
[+0.0013, +0.0455] with weekly-block resampling. It is functionally a
constant forecast correction. The full two-weight edit improved development
performance but worsened Soulis test MAE by 0.199 °C and CYYZ test MAE by
0.591 °C. The half-strength version helped Soulis but hurt CYYZ. These
intervals are exploratory and do not adjust for trying several candidate
edits.

## Conditional cooling repair: fresh 2026 test

We then tested a narrower claim: can one internal weight repair a known failure
across a new period? On CYYZ 2024, the 12-hour prediction was too warm after
strong recent cooling. The condition was fixed as `temperature at issue time −
temperature 24 hours earlier < −5°C`, which uses only past readings. Of the
1,024 features entering the 12-hour output head, feature 916 was selected from
2024 because it tracked that cooling trend and was relatively quiet on other
days. This was exploratory feature screening; the index was not a semantic
property known in advance.

The 2024 examples fitted the size of a change to that one head weight, with a
penalty for errors outside the cooling condition. CYYZ 2025 selected a
shrinkage factor from {0, 0.25, 0.5, 0.75, 1}. The final weight change is
`+0.5025`. We applied it to the checkpoint and saved the state dict under
ignored `.cache/weight_space_pilot/ttm_conditional_weight_edit.pt`. The same
train/validation protocol selected three one-number controls: a correction
only on cooling days, a correction proportional to cooling beyond the −5°C
threshold, and an unconditional bias. The 2026 observations were first scored
after these choices were fixed. The validation loader stops at 2025, so it
cannot use 2026 outcomes or eligibility information.

The available 2026 CYYZ file supports 204 daily origins, January 1–July 23;
27 meet the cooling condition. Scores below are **12-hour MAE in °C**.

| 2026 CYYZ | Cooling, n=27 | Other, n=177 | All, n=204 |
|---|---:|---:|---:|
| Original TTM | 3.103 | 2.654 | 2.713 |
| One-weight checkpoint edit | 2.896 | 2.684 | 2.712 |
| Conditional output correction | **2.817** | **2.654** | **2.676** |
| Cooling-trend output correction | 2.904 | **2.654** | 2.687 |
| Global output bias | 3.044 | 2.668 | 2.718 |

With weekly-block bootstrap resampling, the one-weight edit's cooling-case
gain over the original is +0.207°C [−0.089, +0.512] and its all-case gain is
+0.002°C [−0.095, +0.094]. The conditional output control's gains are
+0.286°C [+0.109, +0.476] and +0.038°C [+0.013, +0.065], respectively.
The direct paired difference between that control and the weight edit is
+0.078°C on cooling cases [−0.153, +0.295] and +0.036°C overall
[−0.053, +0.126]. The intervals do not adjust for the exploratory feature
search or for comparing multiple methods.

After the first 2026 scoring run, we noticed that the validation loader had
allowed one extra December 2025 origin by checking whether a January 2026
24-hour target existed. We restricted validation to observations through 2025
and reran the fixed protocol. The selected coefficients, shrinkage factors,
and 2026 scores did not change; validation has 364 rather than 365 origins.

The weight edit reduced the targeted error, but its effect spilled into other
weather regimes. The simpler output rule localized the correction and gave
the better overall result. Fitting the magnitude of this single weight on
2024 is itself a one-parameter form of head fine-tuning; an ordinary optimizer
on the same fixed coordinate would have the same predictions. This experiment
does not establish that natural-language reasoning found a better update than
standard optimization.

## Interpretation

Direct checkpoint editing works mechanically, and a sparse edit can improve a
local held-out score. These results do **not** establish an LLM advantage:
the most reliable edits behave like simple calibration, and the more targeted
GFS edit overfit the summer failure. The Soulis-trained GFS model is worse
than raw GFS at CYYZ. The 2026 conditional experiment adds evidence that a
single internal weight can respond to a diagnosed weather pattern, but it
also shows why localization matters. A next test should target a model whose
internal representation can gate an update by weather regime, then compare
the proposed edit with matched gradient-based tuning and output corrections
on a second untouched station and time period.

## Reproduce

The run used Python with NumPy 2.4.6, PyTorch 2.13.0, Transformers 5.13.1,
and IBM's [granite-tsfm](https://github.com/ibm-granite/granite-tsfm) at commit
`31fa2f01573341f17e9997280f61481de18ce28c`. TinyTimeMixer weights are
pinned to Hugging Face revision `f04bebdd4c13475b006ce72672e74c9dc28871dc`
(a 3.2 MB download on first run). The scripts use existing repository data;
no bulk data were downloaded or committed.

From the repository root, with `granite-tsfm` importable on `PYTHONPATH`:

```bash
PYTHONPATH=/path/to/granite-tsfm python3 models/weight_space_pilot/run.py 2024 --save
PYTHONPATH=/path/to/granite-tsfm python3 models/weight_space_pilot/run.py 2025 --save
PYTHONPATH=/path/to/granite-tsfm python3 models/weight_space_pilot/run.py 2025 --station cyyz --save
PYTHONPATH=/path/to/granite-tsfm python3 models/weight_space_pilot/conditional_repair.py
python3 models/weight_space_pilot/gfs_repair.py diagnose
python3 models/weight_space_pilot/gfs_repair.py test
python3 models/weight_space_pilot/gfs_repair.py test --station cyyz
python3 models/weight_space_pilot/uncertainty.py .cache/weight_space_pilot/baseline_2025.npz
```

Generated checkpoints and prediction caches are under ignored `.cache/`.
Soulis raw HOBO times are UTC−5 year-round and are shifted +5 hours to UTC;
CYYZ uses ECCC `UTC_DATE`. For TinyTimeMixer, only past-context gaps are
interpolated, at most 12 of 512 hours; missing forecast targets are excluded.
