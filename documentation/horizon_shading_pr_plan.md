# Phased Implementation & Pull Request Plan: Horizon Shading Integration

## Executive Overview

This plan details a phased, low-risk implementation strategy to integrate **360° physical soft horizon shading** into [Helios Forecast](https://github.com/ReikanYsora/Helios-Forecast).

This plan directly incorporates the upstream feature request for manual horizon profile input ([Issue #85](https://github.com/ReikanYsora/Helios-Forecast/issues/85)) and structures the integration into small, logical, independent Pull Requests (PRs) that are easy to review and test.

---

## Technical Architectural Decisions

### 1. Sector Granularity: 24 Sectors (15° Width)
* **Azimuth Resolution**: 24 sectors covering 360° ($15^\circ$ per sector).
* **Rationale**:
  - Home Assistant long-term statistics track energy production in **1-hour buckets**. During daytime, the sun traverses roughly $15^\circ$ of azimuth per hour. A 24-sector grid aligns perfectly with hourly statistical feedback.
  - Compared to a 36-sector grid ($10^\circ$), 24 sectors provide 33% more clear-sky training samples per bin over a 60-day window, preventing under-constrained bins, reducing overfitting risk, and avoiding ~50% extra CPU overhead in the L-BFGS-B optimizer.
  - Smooth continuous transitions across sector boundaries are guaranteed via **circular linear interpolation** with periodic boundary conditions ($360^\circ \equiv 0^\circ$).

### 2. Zero-Disruption & Physical Transposition Math
* **Sigmoid Beam Attenuation Factor**:
  $$S(\alpha, \phi) = \frac{1}{1 + \exp\left(-\frac{\alpha(t) - h(\phi(t))}{w}\right)}$$
  Where $\alpha(t)$ is solar altitude, $\phi(t)$ is solar azimuth, $h(\phi)$ is interpolated obstacle elevation, and $w$ is the transition softness width (degrees).
* **Zero-Disruption Guarantee**:
  When `horizon_profile` is uninitialized or all $0.0^\circ$, $S(\alpha, \phi) \equiv 1.0$ identically across all solar positions.
* **Beam vs. Diffuse Separation**:
  $$I_{\text{direct, POA}} = I_{\text{GHI, eff}} \cdot f_{\text{direct}} \cdot R_b \cdot \text{IAM}(\theta) \cdot S(\alpha, \phi)$$
  Skylight diffuse ($I_{\text{diffuse, POA}}$) and ground-reflected ($I_{\text{ground, POA}}$) irradiance remain unattenuated.

### 3. Optimizer Initialization & Prior Loss Regularization
* **Non-Zero Gradient Initialization**:
  To prevent gradient vanishing during optimization (where $S \approx 1$ when the sun is far above $0^\circ$), initial optimizer parameters start at $h_i = 15.0^\circ$ and $w = 8.0^\circ$.
* **Loss Function with Regularization**:
  $$\text{Loss} = \text{MSE} + \lambda_{\text{smooth}} \sum_{i=1}^{24} (h_{i+1} - h_i)^2 + \lambda_{\text{flat}} \sum_{i=1}^{24} h_i^2$$
  Where $\lambda_{\text{smooth}} = 3.0$ penalizes sharp adjacent sector jumps, and $\lambda_{\text{flat}} = 5.0$ acts as a "no shading by default" prior, pulling unshaded sectors back down to $0.0^\circ$.

---

## Phased Pull Request Breakdown

```
┌───────────────────────────────────────────────────────────────────────────┐
│ PR 1: Horizon Physics Core & Manual Profile Support (Issue #85)           │
│  - Adds `solar/horizon.py` (interpolation & beam attenuation math)         │
│  - Adds manual horizon input to options/config flow (`CONF_HORIZON_PROFILE`)│
│  - Updates `solar/irradiance.py` & `solar/power.py` forward transposition │
│  - 100% zero regression when profile is default/zeroes                   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ PR 2: Optional Horizon Profile Optimizer (`CONF_LEARN_HORIZON_PROFILE`)   │
│  - Adds optional toggle: "Automatically learn horizon profile" (bool)      │
│  - Adds L-BFGS-B optimizer worker (`fit_horizon_profile`)                 │
│  - Integrates background optimization step into `coordinator.py`          │
│  - When disabled, runs traditional empirical ratios over manual horizon   │
└─────────────────────────────────────┬─────────────────────────────────────┘
                                      │
                                      ▼
┌───────────────────────────────────────────────────────────────────────────┐
│ PR 3: Synthetic Benchmarks, Diagnostics & Verification                    │
│  - Synthetic test benchmarks (manual horizon vs. learned horizon)         │
│  - Diagnostics integration (exposing manual vs learned profile vectors)   │
└───────────────────────────────────────────────────────────────────────────┘
```

---

### PR 1: Horizon Physics Core & Manual Horizon Profile Support

#### Objective
Fulfill upstream [Issue #85](https://github.com/ReikanYsora/Helios-Forecast/issues/85) by allowing users to manually configure a 360° horizon profile (or paste comma-separated elevation angles from tools like PVGIS, Solcast, or SunSeeker). The forward model applies this fixed profile to attenuate direct beam irradiance.

#### File Scope
* `custom_components/helios_forecast/solar/horizon.py` *(New Pure Module)*
* `custom_components/helios_forecast/solar/irradiance.py`
* `custom_components/helios_forecast/solar/power.py`
* `custom_components/helios_forecast/config.py`
* `custom_components/helios_forecast/config_flow.py`
* `tests/test_horizon.py` *(New Test File)*
* `tests/test_irradiance.py`, `tests/test_power.py`, `tests/test_config.py`

#### Detailed Changes
1. **`solar/horizon.py`**:
   - `interpolate_horizon_elevation(azimuth_deg, profile)` with circular boundary wrapping ($360^\circ \equiv 0^\circ$).
   - `beam_shading_factor(solar_alt, solar_az, profile, width)`.
   - String parsing helper `parse_horizon_string("0,0,5,10,15,...")` supporting 24 values or comma-separated lists.
2. **`config.py` & `config_flow.py`**:
   - Add `CONF_HORIZON_PROFILE = "horizon_profile"` per panel line in `CONF_ARRAYS` (defaulting to `[0.0] * 24`).
   - Add optional text/number selector input in `config_flow.py` allowing manual horizon elevation entry.
3. **`solar/irradiance.py` & `solar/power.py`**:
   - Add `horizon_profile` field to `PanelOrientation` and `PvLayout`.
   - In `compute_pv_power`, apply `beam_shading_factor` to `direct_poa`.
   - Leave `diffuse_poa` and `ground_poa` unattenuated.

#### Key Verifications & Tests
* **Neutral Horizon Equivalence Test**: Verify all-zero profile (`[0.0] * 24`) produces mathematically identical output ($< 10^{-6}$ relative tolerance) to baseline across all 8,760 hours of a year.
* **Manual Input Test**: Verify manual obstruction profiles (e.g., $20^\circ$ at $90^\circ$ azimuth) attenuate direct morning beam while leaving midday output and overcast diffuse output intact.

---

### PR 2: Optional Horizon Profile Optimizer (`CONF_LEARN_HORIZON_PROFILE`)

#### Objective
Add automatic horizon profile learning as an optional feature. Users can toggle whether Helios should automatically fit/refine the horizon profile from historical recorder data or lock their manual profile as a static physical filter.

#### File Scope
* `custom_components/helios_forecast/solar/horizon.py` *(Append optimizer)*
* `custom_components/helios_forecast/config.py`
* `custom_components/helios_forecast/config_flow.py`
* `custom_components/helios_forecast/coordinator.py`
* `tests/test_horizon_fitting.py` *(New Test File)*

#### Detailed Changes
1. **`config.py` & `config_flow.py`**:
   - Add `CONF_LEARN_HORIZON_PROFILE = "learn_horizon_profile"` boolean setting (default: `True` if production entity is present, otherwise `False`).
2. **`solar/horizon.py`**:
   - Implement `fit_horizon_profile(production_buckets, weather, layout, ...)` using `scipy.optimize.minimize(method='L-BFGS-B')`.
   - Includes alive initialization ($h_i = 15^\circ, w = 8^\circ$) and priors ($\lambda_{\text{smooth}}=3.0$, $\lambda_{\text{flat}}=5.0$).
3. **`coordinator.py`**:
   - When `CONF_LEARN_HORIZON_PROFILE` is `True` and production history is available:
     - Run `fit_horizon_profile()` in `hass.async_add_executor_job`.
     - Atomically save updated `horizon_profile` back to config entry storage.
     - Sequence horizon fitting **before** `build_sky_residual_map` and `build_library`, so residual weather ratios are learned on top of the physical horizon-shaded baseline.
   - When `CONF_LEARN_HORIZON_PROFILE` is `False`:
     - Skip horizon optimizer. Use user-supplied static `horizon_profile` for transposition and run standard empirical ratio/analog learning directly on top of it.

#### Key Verifications & Tests
* Verify manual profile is preserved without being overwritten when `learn_horizon_profile=False`.
* Verify L-BFGS-B optimizer successfully fits horizon sector elevations from historical uncurtailed daytime production when `learn_horizon_profile=True`.

---

### PR 3: Synthetic Benchmarks, Diagnostics & Documentation

#### Objective
Provide diagnostic tools, synthetic unit test benchmarks, and documentation updates for upstream integration.

#### File Scope
* `custom_components/helios_forecast/diagnostics.py`
* `tests/test_benchmark.py`
* `README.md`

#### Detailed Changes
1. **`diagnostics.py`**:
   - Include `horizon_profile`, `horizon_source` (`"manual"` vs `"learned"`), and transition softness `w` in HA diagnostics JSON.
2. **`tests/test_benchmark.py`**:
   - **Benchmark A (Unshaded Control)**: Optimizer on unshaded data returns $h_i \approx 0.0^\circ$.
   - **Benchmark B (Eastern Tree Obstruction)**: Morning shading data correctly recovers $h_i \approx 20^\circ$ for $80^\circ - 120^\circ$ azimuths.
   - **Benchmark C (Seasonal Generalization)**: Summer-trained model retains high accuracy on winter sun paths.
3. **`README.md`**:
   - Document manual horizon configuration (importing from PVGIS/SunSeeker) and automatic profile learning.

---

## Summary Comparison Matrix

| PR | Primary Feature Added | Upstream Feature / Goal Addressed | Regression Risk |
| :--- | :--- | :--- | :--- |
| **PR 1** | Horizon Physics Engine & Manual Profile Input | Fulfills [Issue #85](https://github.com/ReikanYsora/Helios-Forecast/issues/85) (User-provided horizon) | Zero (Neutral profile equals baseline) |
| **PR 2** | Optional L-BFGS-B Horizon Learning | Automatic 360° shading profile optimization | Low (Disabled by toggle or when history absent) |
| **PR 3** | Benchmarks, Diagnostics & Documentation | Verification suite, diagnostics transparency, README | None |
