Name(s): Tanisha

# **Topic Summary:**

### TLDR;

> downscaling = super-resolution from computer vision applied to weather → map a coarse field to a fine one, way cheaper than running a high-res physical model.
> 
> the catch: weather isn't just an image. plain CNNs blur (they average over the many plausible fine-scale outcomes), so the field moved CNN → GAN → diffusion to get sharp, realistic, probabilistic output. 
> 
> - terrain, lake/land masks, and station obs matter as extra inputs, and you can't judge results on RMSE alone (it rewards blur) ⇒ you need spectra + CRPS + calibration checks.
> 
> for us: our layer is basically MOS-style post-processing → take a coarse ERA5 forecast, correct + sharpen it toward local KW reality, ideally anchored to station obs.

### If you have time to read:

Essentially, downscaling and super resolution are tied to one another (note that super resolution is an AI and machine learning technique that intelligently increases an image or video's resolution by adding missing details)

- Both map a low-resolution (LR) field to a high-resolution (HR) version of the same field, so image-SR architectures directly translate to climate/weather data.
- The appeal of using downscaling is speed; we are able to get fine-scale structure that interpolation misses → this is way cheaper than running a high-res physical model.

Since weather data isn’t just an image, plain CNN super-resolution just oversmooths and regresses to the mean ⇒ newer work adds physics constraints, observational guidance, and generative architectures to improve the output. 

- Weather data carries the following constraints → physical constraints (ie. mass, energy consistency), terrain, season, water bodies + interesting behaviour that lies in the extremes

### Layer Decision

We also have to consider where this layer sits… From the research (see arxiv src 5 below), we basically have two choices:

1. Perfect Prognosis (PP)
    1. learns LR→HR from observations
2. Model Output Statistics (MOS)
    1. learns model output→observations
    2. folds in bias correction
    3. basically: take a coarse forecast, correct+sharpen towards local KW reality

### Methods to adopt

1. CNN (deterministic) - stacked SRCNN for precipitation downscaling
    1. UNets and residual nets beat naive disaggregation
    2. Weakness: pixel-wise loss == blurr
    3. Src 1
2. GAN (sharp) - generate ensembles with roughly correct variability, checked via rank stats
    1. They were able to super-res wind/solar up to 50x with turbulence-consistent output ⭐
    2. Weakness: hallucination, training inconsistencies
    3. Src 3
3. Diffusion (frontier) - downscale 25km → 2km via two-step UNet mean and diff residual
    1. CorrDiff downscaled it - framed as ML post-processing of coarse predictions ⇒ can synthesize new channels
    2. Weakness: super expensive and hard to calibrate
    3. Src 4

# **Key Questions:**

- What are our LR→HR pairs?
    - Coarse ERA5 (we have Z500 plumbing) → station obs, high-res reanalysis, or radar? Decides PP vs MOS
- Deterministic or probabilistic: do we actually need ensembles/uncertainty, or is one sharp estimate enough?
- Which static predictors do we have for KW? Lake mask + land use probably > elevation here… (?)
- What metrics go in the AutoML objective? Pure RMSE → search picks the blurriest model. Add spectral term / CRPS?
- Which physics/statistical constraints should the layer enforce so output is both sharp and meteorologically trustworthy?

# **Sources:**

1. **[DeepSD: Generating High Resolution Climate Change Projections through Single Image Super-Resolution]** (https://dl.acm.org/doi/10.1145/3097983.3098004)
    - the origin point for CV-downscaling - stacked SRCNN framework for statistical downscaling of precipitation
    - establishes the "downscaling = SR" framing everyone cites
    - good for intro/background section
2. **[Adversarial super-resolution of climatological wind and solar data** (https://www.pnas.org/doi/10.1073/pnas.1918964117)
    - canonical GAN. up to 50×, correct small-scale turbulence stats. code: NREL/PhIRE
3. **[Stochastic Super-Resolution for Downscaling Time-Evolving Atmospheric Fields with a GAN]** (https://arxiv.org/abs/2005.10374)
4. **[Residual Corrective Diffusion Modeling for Km-scale Atmospheric Downscaling (CorrDiff)]** (https://arxiv.org/abs/2309.15214)
    - closest published analog to us!!!
    - 25 km ERA5 → 2km, UNet mean + diffusion residual, framed as post-processing
5. **[Hard-Constrained Deep Learning for Climate Downscaling]** (https://arxiv.org/abs/2208.05424)
    - SR equivalence - research more into this
6. **[Observation-Guided Meteorological Field Downscaling at Station Scale: A Benchmark and a New Method]** (https://arxiv.org/abs/2401.11960)
    1. corresponding lit review - https://www.themoonlight.io/en/review/observation-guided-meteorological-field-downscaling-at-station-scale-a-benchmark-and-a-new-method
    2. this is directly relevant to KW - argues that image-SR framing biases output away from station observations, anchor fine-scale pred to obs instead
    3. benchmark+dataset ⭐
