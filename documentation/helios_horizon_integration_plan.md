# Project Plan: Physical Horizon Shading Integration & Weather Provider Strategy for Helios Forecast

---

## Executive Summary

**Helios Forecast** (`ReikanYsora/Helios-Forecast`) is a mature, self-learning Home Assistant integration that predicts photovoltaic (PV) generation and projects battery state of charge. Its present learning engine transposes Open-Meteo horizontal irradiance to panel geometry and applies an **empirical/analog ratio lookup library** to correct predictions against historical weather conditions.

While effective for atmospheric variations, pure empirical ratio matching suffers in complex shading environments (e.g. trees, nearby roofs, chimneys) because:
1. It does not generalize shading across seasons (summer vs. winter sun elevation paths).
2. It conflates fixed physical obstructions with transient weather cover.
3. It often drops predicted power to near zero when shaded, neglecting diffuse skylight.

This project plan outlines a **non-disruptive strategy to integrate explicit physical 360° soft horizon shading profile learning into Helios Forecast**, along with an in-depth **comparative evaluation of Open-Meteo vs. DWD (Deutscher Wetterdienst) MOSMIX weather forecast providers**, and a **robust regression testing methodology** to ensure prediction accuracy does not regress.

---

## Part 1: Weather Provider Analysis — Open-Meteo vs. DWD MOSMIX

### 1.1 Overview of Weather Forecast Sources

| Attribute | **Open-Meteo API** | **DWD MOSMIX** |
| :--- | :--- | :--- |
| **Geographic Coverage** | **Global** (Any latitude/longitude worldwide) | **Regional / Station-based** (~5,400 station points globally, dense in Central Europe) |
| **Underlying Models** | Multi-model integration: ECMWF, GFS, ICON, Arpege, MET Norway | DWD ICON global & regional models post-processed via Model Output Statistics (MOS) |
| **PV Radiation Variables** | $GHI$ (Global Horizontal), $DNI$ (Direct Normal), $DHI$ (Diffuse Horizontal) | $GHI$ ($kJ/m^2 \rightarrow W/m^2$), Effective Cloud Cover ($N$), 2m Air Temp |
| **Snow Depth / Snow Cover** | **Supported directly** (`snow_depth` in meters) | Not standard in MOSMIX point forecasts |
| **Update Frequency** | Hourly / 3-hourly | 4 times daily (00, 06, 12, 18 UTC) |
| **Access & Rate Limits** | REST API (No key required for non-commercial <10,000 daily calls) | Free public KML/JSON downloads from DWD Open Data server |

---

### 1.2 Can Open-Meteo Fulfill All PV Forecast Requirements?

**Yes.** Open-Meteo is fully capable of serving as the primary weather provider for a production-grade PV forecast tool:

- **Direct Radiation Decomposition**: Open-Meteo provides direct separation of $GHI$, $DNI$, and $DHI$, eliminating the need for empirical decomposition algorithms (such as the Erbs model) and providing immediate beam/diffuse components for geometric transposition.
- **Global Usability**: Unlike nation-specific meteorological services, Open-Meteo works seamlessly anywhere in the world without user configuration of station IDs.
- **Snow Cover Integration**: Open-Meteo includes explicit hourly snow depth forecasts, which are crucial for modeling winter panel coverage on tilted roofs.

---

### 1.3 Does DWD MOSMIX Add Value Where Available?

**Yes, as a regional augmentation layer for Central Europe.** DWD MOSMIX provides distinct advantages over raw Numerical Weather Prediction (NWP) models due to its station-level statistical post-processing (Model Output Statistics):

1. **Station-Level Calibration (MOS)**: MOSMIX applies statistical regression equations derived from years of ground-station measurements. This corrects systematic local model biases (e.g. valley fog, microclimate cloud development, coastal humidity).
2. **Superior Local Cloud Cover & Irradiance Tuning**: Studies across Germany and Central Europe consistently show that DWD MOSMIX achieves lower Root Mean Square Error (RMSE) for 24h–72h GHI point forecasts than uncalibrated grid-based NWP models.

#### Recommended Hybrid Weather Strategy for Helios Forecast

Open-Meteo remains the foundational, zero-configuration weather provider for all worldwide installations. For users in Central Europe, DWD MOSMIX can be enabled as an **augmentation layer**:

- **Open-Meteo** provides the 3D physical atmospheric beam-to-diffuse split ratio $f_{\text{diffuse}} = DHI_{\text{om}} / GHI_{\text{om}}$ and hourly `snow_depth`.
- **DWD MOSMIX** provides ground-station calibrated total $GHI_{\text{dwd}}$, 2m temperature, and effective cloud cover.
- **Augmented Irradiance Components**:
  $$DHI_{\text{augmented}} = GHI_{\text{dwd}} \cdot f_{\text{diffuse}}$$
  $$DNI_{\text{augmented}} = GHI_{\text{dwd}} \cdot \frac{1 - f_{\text{diffuse}}}{\cos(\theta_z)}$$

This hybrid approach combines DWD station accuracy with Open-Meteo physical radiation splitting without needing empirical Erbs decomposition algorithms.

---

## Part 2: Core Horizon Shading Architecture

### 2.1 Physics Formulation

Physical horizon obstacles are parameterized as a vector of $N$ angular elevation thresholds across 360° of azimuth:

$$H = [h_1, h_2, \dots, h_N] \quad \text{where } h_i \text{ is the obstacle elevation angle in degrees for sector } i$$

For $N = 24$ sectors, each sector covers $15^\circ$ of azimuth $\phi$.

#### Differentiable Sigmoid Beam Attenuation
To allow smooth gradient-based parameter optimization (e.g. L-BFGS-B), the binary obstacle cutoff is approximated using a continuous sigmoid factor $S(\alpha, \phi)$:

$$S(\alpha, \phi) = \frac{1}{1 + \exp\left( -k \cdot (\alpha(t) - h(\phi(t))) \right)}$$

Where:
- $\alpha(t)$ is the solar elevation angle at time $t$.
- $\phi(t)$ is the solar azimuth angle at time $t$.
- $h(\phi(t))$ is the interpolated horizon obstacle height at azimuth $\phi(t)$.
- $k$ is the sharpness parameter governing the transition (typically $k \approx 0.5 - 1.0$).

#### Beam vs. Diffuse Light Separation
The horizon shading factor $S(\alpha, \phi)$ attenuates **Direct Normal Irradiance ($DNI$)**, while **Diffuse Horizontal Irradiance ($DHI$)** is preserved (or scaled by an isotropic sky view factor):

$$I_{\text{transposed}}(t) = DNI(t) \cdot S(\alpha, \phi) \cdot \cos(\theta(t)) + DHI(t) \cdot \text{SVF}(\text{tilt})$$

Where $\theta(t)$ is the angle of incidence on the panel, and $\text{SVF}$ is the sky view factor.

---

### 2.2 Why This Complements Helios's Existing Empirical Ratios

Integrating physical horizon shading creates a two-stage hybrid prediction engine:

```
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    Open-Meteo Base + Optional DWD Augmentation                  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                   Stage 1: Physical Transposition & Horizon Filter               │
│   - Solar position geometry (Zenith, Azimuth, Incidence Angle)                  │
│   - Beam DNI attenuated by learned continuous 360° horizon profile h(φ)         │
│   - Skylight diffuse DHI preserved                                               │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                    Stage 2: Helios Empirical Ratio Library                       │
│   - Matches atmospheric condition buckets (clouds, temp, humidity)              │
│   - Corrects residual weather errors without conflating them with fixed shading  │
└────────────────────────────────────────┬─────────────────────────────────────────┘
                                         │
                                         ▼
┌──────────────────────────────────────────────────────────────────────────────────┐
│                             Final Power & Energy Forecast                        │
│   - 15-minute time series, P10/P90 confidence bounds, battery SoC projection    │
└──────────────────────────────────────────────────────────────────────────────────┘
```

---

## Part 3: Step-by-Step Implementation Plan for Helios Horizon Integration

> **Main Objective**: Add physical 360° horizon obstacle learning into Helios Forecast without disrupting existing features, configuration flows, or Home Assistant entity schemas.

### Phase 1: Horizon Physics Engine & Forward Model Integration

- **Task 1.1: Horizon Profile Data Structure & Module Location**
  - Add a dedicated pure physics module `custom_components/helios_forecast/solar/horizon.py` holding `HorizonProfile` (24 sectors of $15^\circ$), circular azimuth interpolation, and `beam_shading_factor(solar_alt, solar_az, horizon_profile, width)`.
  - Add an internal 24-element list `horizon_profile` (defaulting to all `0.0` degrees) to `PanelOrientation` in `solar/irradiance.py` and `PvLayout` in `solar/power.py`.
  - Backward compatibility: If loading existing saved configuration JSON where `horizon_profile` is missing, initialize it to `[0.0] * 24`.

- **Task 1.2: Transposition Enhancement (`solar/irradiance.py`)**
  - In `compute_pv_power`, compute solar altitude $\alpha$ and azimuth $\phi$ via `sun_position`.
  - Compute continuous beam attenuation factor $S(\alpha, \phi) = \text{beam\_shading\_factor}(\alpha, \phi, \text{horizon\_profile}, w)$.
  - Apply attenuation to direct beam Plane-of-Array irradiance:
    $$\text{direct\_poa} = \text{ghi\_eff} \cdot \text{direct\_fraction} \cdot R_b \cdot \text{IAM} \cdot S(\alpha, \phi)$$
  - Keep skylight diffuse ($\text{diffuse\_poa}$) and ground-reflected ($\text{ground\_poa}$) irradiance intact.
  - **Zero Disruption Guarantee**: When $h(\phi) = 0.0^\circ$, $S(\alpha, \phi) = 1.0$, producing mathematically identical output to Helios's current baseline transposition.

---

### Phase 2: Asynchronous Horizon Profile Optimizer

- **Task 2.1: Data Preparation & Curtailment Exclusion**
  - Query historical production ($kWh$) and historical Open-Meteo weather matching the training window (rolling 60 days).
  - **Curtailment Filter**: Exclude hours where curtailment entities are active or inverter AC power limits are reached (reusing `curtailment.py`).

- **Task 2.2: L-BFGS-B Horizon Fitting (`solar/horizon.py` & `coordinator.py`)**
  - Implement optimizer worker `fit_horizon_profile(production_buckets, weather, layout, ...)` using `scipy.optimize.minimize(method='L-BFGS-B')`.
  - **Non-Zero Gradient Alive Initialization**: Initialize optimization parameters at $h_i = 15.0^\circ$ and $w = 8.0^\circ$ (matching `pySolarForecast`) so initial predictions sit within the active sun elevation region and numerical gradients $\frac{\partial \text{Loss}}{\partial h_i}$ are non-zero.
  - **Loss Function with Priors**: Mean Squared Error (MSE) between actual PV generation and model prediction across uncurtailed daytime hours, plus smoothness prior $\lambda_{\text{smooth}} \sum (h_{i+1} - h_i)^2$ ($\lambda_{\text{smooth}} = 3.0$) and flatness prior $\lambda_{\text{flat}} \sum h_i^2$ ($\lambda_{\text{flat}} = 5.0$).
  - **Flatness Prior Function**: Ensures unshaded azimuth sectors actively converge to $0.0^\circ$ obstacle height.
  - Bounds: $0^\circ \le h_i \le 60^\circ$ for sector heights, $0.5^\circ \le w \le 20.0^\circ$ for transition width.
  - Execute fitting within non-blocking thread pools (`hass.async_add_executor_job`).

- **Task 2.3: Integration into Coordinator Refresh Sequence**
  - Sequence in `coordinator.py`: Horizon fitting runs **before** `build_sky_residual_map` and `build_library`, so residual weather errors are learned on top of the physical horizon-shaded baseline.
  - Atomically save updated `horizon_profile` values to Helios's persistent storage (`/config/.storage/helios_forecast...`).

---

### Phase 3: Diagnostics & Pull Request Delivery

- **Task 3.1: Diagnostics Integration**
  - Expose learned 360° horizon profile values in Helios's Home Assistant diagnostics output (`diagnostics.json`).

- **Task 3.2: Community Pull Request Submission**
  - Submit pull request to `ReikanYsora/Helios-Forecast` maintaining 100% compliance with Helios's data contracts, GPL-3.0 license, and HACS distribution requirements.

---

## Part 4: Regression Testing & Validation Strategy

To guarantee that adding physical horizon shading introduces **zero regressions** and ensures predictions do not get worse for any installation:

### 4.1 Automated Unit & Mathematical Verification
- **Neutral Horizon Equivalence Test**: Verify that when `horizon_profile = [0.0] * 24`, the new forward model generates mathematically identical predictions (within $10^{-6}$ relative tolerance) to the current Helios baseline across all solar elevations and azimuths.
- **Sigmoid Gradient Non-Saturation Test**: Verify that initial parameter bounds and sigmoid softness $k$ yield non-zero numerical gradients $\partial \text{Loss} / \partial h_i > 10^{-4}$ during initial L-BFGS-B optimization steps.
- **Night-Filter Edge Case Test**: Verify that power predictions forcedly evaluate to $0.0\,\text{W}$ whenever solar elevation $\alpha \le 0^\circ$ (zenith $\ge 90^\circ$).

### 4.2 Synthetic Benchmark Dataset Suite
Create a test suite consisting of three simulated installation environments:

1. **Benchmark A: Unshaded Roof (Control / Zero-Regression Guard)**
   - *Setup*: Completely open horizon with no trees or obstructions.
   - *Expectation*: Horizon optimizer outputs $h_i \approx 0^\circ$ across all sectors (enforced by the flatness prior $\lambda_{\text{flat}}$).
   - *Acceptance Criteria*: RMSE between baseline Helios and Horizon-enabled Helios must differ by $< 1\%$.
2. **Benchmark B: Eastern Tree Obstruction (15°–25° elevation between 80°–120° azimuth)**
   - *Setup*: Shading obstacle in eastern morning sky.
   - *Expectation*: Optimizer fits $h_i \approx 20^\circ$ in eastern sectors.
   - *Acceptance Criteria*: RMSE during morning hours (06:00–10:00) decreases by $> 30\%$ compared to baseline.
3. **Benchmark C: Seasonal Generalization (Summer Training $\rightarrow$ Winter Evaluation)**
   - *Setup*: Train model using only June–August data. Evaluate on December–January data.
   - *Expectation*: Horizon-enabled model retains high prediction accuracy in winter by applying the learned summer horizon geometry $h(\phi)$ to winter sun paths. Baseline empirical model exhibits higher error due to unlearned winter weather-ratio buckets.

### 4.3 Quantitative Accuracy Metrics

Evaluations must measure the following statistical metrics across test suites:

- **Root Mean Square Error (RMSE)**:
  $$\text{RMSE} = \sqrt{\frac{1}{N} \sum_{t=1}^N \left( \hat{P}(t) - P_{\text{actual}}(t) \right)^2}$$
- **Coefficient of Determination ($R^2$)**:
  $$R^2 = 1 - \frac{\sum \left( P_{\text{actual}}(t) - \hat{P}(t) \right)^2}{\sum \left( P_{\text{actual}}(t) - \bar{P} \right)^2}$$
- **Mean Absolute Shading Error (MASE during shaded windows)**: Evaluated specifically during hours where solar elevation $\alpha(t) < h(\phi(t))$.

---

## Appendix: Additional Potential PySolarForecast Contributions for Helios

Beyond 360° physical horizon shading (the core subject of this plan), `pySolarForecast` contains several physics formulations and tooling capabilities that could form valuable future pull requests to `Helios Forecast`:

### Appendix A: DWD MOSMIX Regional Weather Augmentation Layer
- **Concept**: Add an optional "Weather Provider" dropdown in Helios options flow allowing users in Central Europe to select DWD MOSMIX as an augmentation source.
- **Benefit**: Scales Open-Meteo's native beam/diffuse ratio $f_{\text{diffuse}}$ by DWD MOSMIX station-calibrated $GHI_{\text{dwd}}$, reducing 24h–72h prediction error in Germany, Austria, Switzerland, and neighboring regions without requiring empirical Erbs decomposition.

### Appendix B: Vertical & Facade Panel Support with String-Specific Snow Attenuation
- **Concept**: Model snow depth attenuation using string-specific sigmoid functions:
  $$\text{Snow Factor} = \frac{1}{1 + \exp\left( \frac{\text{snow\_depth} - \text{snow\_threshold}}{\text{snow\_width}} \right)}$$
- **Vertical/Facade Benefit**: For vertical wall-mounted panels (90° tilt, e.g. facade strings), snow does not stick or accumulate. The optimizer naturally fits a high `snow_threshold` (or zero attenuation) for vertical strings so wall panels continue predicting production during winter while roof panels are marked snow-covered.

### Appendix C: Temperature-Dependent Cell Efficiency (NOCT Thermal Derating)
- **Concept**: Panel efficiency drops as cell temperature rises (typically $-0.4\%$ per $^\circ\text{C}$ above $25^\circ\text{C}$). Estimate cell temperature from ambient air temperature and Plane-of-Array (POA) irradiance:
  $$T_{\text{cell}} = T_{\text{air}} + k_{\text{NOCT}} \cdot I_{\text{POA}}$$
  $$\text{Efficiency Factor} = 1 - \gamma \cdot (T_{\text{cell}} - 25^\circ\text{C})$$
- **Benefit**: Prevents over-prediction on hot, clear summer days where panel temperatures exceed $60^\circ\text{C}$.

### Appendix D: Historical Data Pre-loading Script (SQLite / CSV -> HA Statistics)
- **Concept**: Standalone CLI script or service to insert historical hourly PV generation ($kWh$) from `pySolarForecast` SQLite databases or CSV files directly into Home Assistant's internal `statistics` and `statistics_meta` database tables.
- **Benefit**: Bypasses the multi-week warmup period, allowing Helios Forecast to immediately train its horizon profile and empirical ratio library on years of historical data upon installation.
