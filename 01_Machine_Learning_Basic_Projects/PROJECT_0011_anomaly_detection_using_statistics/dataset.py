"""
Project 011: Statistical Anomaly Detection - Benchmark Datasets
===============================================================
A rich collection of realistic industrial telemetry, cybersecurity,
financial transactions, medical vitals, 2D geometric manifolds,
and streaming time-series designed to stress-test statistical detectors.

Part of the 1,000 AI Projects Implementation Library.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, List, Optional


# =============================================================================
# 1. REAL-WORLD INDUSTRIAL & BUSINESS TELEMETRY GENERATORS
# =============================================================================

def generate_financial_fraud_telemetry(
    n_samples: int = 1000,
    contamination: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Simulates e-commerce and credit transaction payment streams.
    Normal transactions have typical spending, low velocity, and nearby geolocations.
    Fraudulent anomalies present as rapid bursts, extreme dollar amounts, and distant geolocations.
    """
    np.random.seed(seed)
    n_anomalies = max(5, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies

    # Normal Transactions
    amount = np.random.lognormal(mean=3.8, sigma=0.6, size=n_normal)  # ~$45 median
    velocity_1h = np.random.poisson(lam=1.5, size=n_normal) + 1
    dist_home = np.random.exponential(scale=12.0, size=n_normal)  # km
    foreign_ip = np.random.beta(a=0.5, b=10.0, size=n_normal)
    decline_rate = np.random.beta(a=0.2, b=15.0, size=n_normal)

    # Injected Anomalous Frauds
    anom_amount = np.concatenate([
        np.random.uniform(1200.0, 4500.0, size=n_anomalies // 2),  # Large wire / cashout
        np.random.uniform(0.10, 0.99, size=n_anomalies - (n_anomalies // 2))  # Micro-testing probe
    ])
    anom_velocity = np.random.randint(8, 25, size=n_anomalies)  # Rapid carding bot burst
    anom_dist = np.random.uniform(450.0, 3500.0, size=n_anomalies)  # Far overseas distance
    anom_foreign = np.random.uniform(0.70, 0.99, size=n_anomalies)
    anom_decline = np.random.uniform(0.40, 0.95, size=n_anomalies)

    df_norm = pd.DataFrame({
        "amount_usd": amount,
        "velocity_1h": velocity_1h,
        "dist_from_home_km": dist_home,
        "foreign_ip_score": foreign_ip,
        "recent_decline_rate": decline_rate,
        "is_anomaly": 0
    })

    df_anom = pd.DataFrame({
        "amount_usd": anom_amount,
        "velocity_1h": anom_velocity,
        "dist_from_home_km": anom_dist,
        "foreign_ip_score": anom_foreign,
        "recent_decline_rate": anom_decline,
        "is_anomaly": 1
    })

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_cols = ["amount_usd", "velocity_1h", "dist_from_home_km", "foreign_ip_score", "recent_decline_rate"]
    return df, feature_cols, "is_anomaly"


def generate_cloud_server_telemetry(
    n_samples: int = 1000,
    contamination: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Simulates high-throughput cloud infrastructure telemetry.
    Normal servers maintain steady utilization, predictable I/O, and low HTTP error rates.
    Anomalies represent memory leaks, CPU thread starvation, deadlock, and HTTP 5xx spikes.
    """
    np.random.seed(seed)
    n_anomalies = max(5, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies

    # Normal System Health
    cpu = np.clip(np.random.normal(loc=42.0, scale=10.0, size=n_normal), 5.0, 80.0)
    mem = np.clip(np.random.normal(loc=55.0, scale=8.0, size=n_normal), 20.0, 78.0)
    disk_iops = np.random.normal(loc=350.0, scale=45.0, size=n_normal)
    latency_ms = np.random.lognormal(mean=2.8, sigma=0.4, size=n_normal)  # ~16-25ms
    err_5xx = np.random.poisson(lam=0.2, size=n_normal)

    # Injected Outages & Starvation
    anom_cpu = np.random.choice([
        np.random.uniform(94.0, 99.9, size=n_anomalies // 2),  # 100% peg
        np.random.uniform(0.5, 3.0, size=n_anomalies - (n_anomalies // 2))  # Frozen zombie worker
    ], size=1)[0]
    anom_mem = np.random.uniform(92.0, 99.5, size=n_anomalies)  # OOM threshold
    anom_iops = np.random.choice([
        np.random.uniform(1200.0, 2500.0, size=n_anomalies // 2),  # Thrashing I/O
        np.random.uniform(0.0, 10.0, size=n_anomalies - (n_anomalies // 2))  # Deadlock
    ], size=1)[0]
    anom_latency = np.random.uniform(450.0, 2800.0, size=n_anomalies)
    anom_err_5xx = np.random.poisson(lam=18.0, size=n_anomalies) + 5

    df_norm = pd.DataFrame({
        "cpu_pct": cpu,
        "memory_pct": mem,
        "disk_iops": disk_iops,
        "latency_ms": latency_ms,
        "http_5xx_rate": err_5xx,
        "is_anomaly": 0
    })

    df_anom = pd.DataFrame({
        "cpu_pct": anom_cpu,
        "memory_pct": anom_mem,
        "disk_iops": anom_iops,
        "latency_ms": anom_latency,
        "http_5xx_rate": anom_err_5xx,
        "is_anomaly": 1
    })

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_cols = ["cpu_pct", "memory_pct", "disk_iops", "latency_ms", "http_5xx_rate"]
    return df, feature_cols, "is_anomaly"


def generate_iot_sensor_telemetry(
    n_samples: int = 1000,
    contamination: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Simulates industrial heavy machinery sensor readings (bearings, turbines, pumps).
    Anomalies flag mechanical wear, casing overheating, and pressure breaches.
    """
    np.random.seed(seed)
    n_anomalies = max(5, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies

    # Normal Machine Operation
    vibration = np.random.normal(loc=1.8, scale=0.35, size=n_normal)  # mm/s RMS
    temp = np.random.normal(loc=68.0, scale=4.0, size=n_normal)  # deg C
    pressure = np.random.normal(loc=140.0, scale=7.0, size=n_normal)  # PSI
    power_kw = np.random.normal(loc=45.0, scale=3.5, size=n_normal)  # kW
    acoustic_db = np.random.normal(loc=58.0, scale=3.0, size=n_normal)  # dB

    # Mechanical Faults & Bearing Seizure
    anom_vib = np.random.uniform(4.2, 9.5, size=n_anomalies)  # Extreme vibration harmonic
    anom_temp = np.random.uniform(88.0, 115.0, size=n_anomalies)  # Thermal runaway
    anom_pressure = np.random.choice([
        np.random.uniform(185.0, 240.0, size=n_anomalies // 2),  # Overpressure
        np.random.uniform(40.0, 75.0, size=n_anomalies - (n_anomalies // 2))  # Hydraulic seal rupture
    ], size=1)[0]
    anom_power = np.random.uniform(65.0, 92.0, size=n_anomalies)
    anom_acoustic = np.random.uniform(76.0, 98.0, size=n_anomalies)

    df_norm = pd.DataFrame({
        "vibration_rms": vibration,
        "casing_temp_c": temp,
        "line_pressure_psi": pressure,
        "power_draw_kw": power_kw,
        "acoustic_noise_db": acoustic_db,
        "is_anomaly": 0
    })

    df_anom = pd.DataFrame({
        "vibration_rms": anom_vib,
        "casing_temp_c": anom_temp,
        "line_pressure_psi": anom_pressure,
        "power_draw_kw": anom_power,
        "acoustic_noise_db": anom_acoustic,
        "is_anomaly": 1
    })

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_cols = ["vibration_rms", "casing_temp_c", "line_pressure_psi", "power_draw_kw", "acoustic_noise_db"]
    return df, feature_cols, "is_anomaly"


def generate_medical_vital_signs(
    n_samples: int = 800,
    contamination: float = 0.06,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Simulates ICU patient telemetry.
    Normal vitals: Resting heart rate, normal blood pressure, SpO2 > 95%.
    Anomalies: Tachycardia, hypoxemia, hypertension crisis, septic shock collapse.
    """
    np.random.seed(seed)
    n_anomalies = max(5, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies

    hr = np.random.normal(loc=72.0, scale=7.0, size=n_normal)
    sys_bp = np.random.normal(loc=120.0, scale=8.0, size=n_normal)
    dia_bp = np.random.normal(loc=78.0, scale=5.0, size=n_normal)
    spo2 = np.clip(np.random.normal(loc=98.2, scale=1.0, size=n_normal), 94.0, 100.0)
    resp = np.random.normal(loc=16.0, scale=2.0, size=n_normal)

    # Injected critical clinical events
    anom_hr = np.random.choice([
        np.random.uniform(145.0, 190.0, size=n_anomalies // 2),  # Severe tachycardia
        np.random.uniform(32.0, 44.0, size=n_anomalies - (n_anomalies // 2))  # Extreme bradycardia
    ], size=1)[0]
    anom_sys = np.random.choice([
        np.random.uniform(185.0, 225.0, size=n_anomalies // 2),  # Hypertensive crisis
        np.random.uniform(60.0, 75.0, size=n_anomalies - (n_anomalies // 2))  # Septic hypotension
    ], size=1)[0]
    anom_dia = np.random.uniform(40.0, 125.0, size=n_anomalies)
    anom_spo2 = np.random.uniform(75.0, 88.0, size=n_anomalies)  # Critical desaturation
    anom_resp = np.random.uniform(30.0, 48.0, size=n_anomalies)  # Tachypnea

    df_norm = pd.DataFrame({
        "heart_rate_bpm": hr,
        "systolic_bp": sys_bp,
        "diastolic_bp": dia_bp,
        "spo2_pct": spo2,
        "respiration_rate": resp,
        "is_anomaly": 0
    })

    df_anom = pd.DataFrame({
        "heart_rate_bpm": anom_hr,
        "systolic_bp": anom_sys,
        "diastolic_bp": anom_dia,
        "spo2_pct": anom_spo2,
        "respiration_rate": anom_resp,
        "is_anomaly": 1
    })

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    feature_cols = ["heart_rate_bpm", "systolic_bp", "diastolic_bp", "spo2_pct", "respiration_rate"]
    return df, feature_cols, "is_anomaly"


# =============================================================================
# 2. 2D GEOMETRIC BENCHMARK MANIFOLDS
# =============================================================================

def generate_correlated_gaussian(
    n_samples: int = 600,
    contamination: float = 0.06,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    2D Correlated Gaussian distribution with strong covariance (r = 0.88).
    Includes 'Off-Diagonal' anomalies: points that have normal 1D values (Z < 2.5)
    but completely violate the correlation structure.
    Perfect demonstration of why Univariate Z-Score fails while Mahalanobis succeeds!
    """
    np.random.seed(seed)
    n_anomalies = max(4, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies

    # Bivariate normal with correlation r = 0.88
    mean = [0.0, 0.0]
    cov = [[1.0, 0.88], [0.88, 1.0]]
    pts_norm = np.random.multivariate_normal(mean, cov, size=n_normal)

    # Injected off-diagonal anomalies (orthogonal to the principal correlation axis)
    # Along y = -x instead of y = x
    anom_x = np.random.uniform(-2.5, 2.5, size=n_anomalies)
    anom_y = -anom_x + np.random.normal(0, 0.2, size=n_anomalies)
    pts_anom = np.column_stack([anom_x, anom_y])

    df_norm = pd.DataFrame(pts_norm, columns=["Feature_X1", "Feature_X2"]).assign(is_anomaly=0)
    df_anom = pd.DataFrame(pts_anom, columns=["Feature_X1", "Feature_X2"]).assign(is_anomaly=1)

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df, ["Feature_X1", "Feature_X2"], "is_anomaly"


def generate_multimodal_clusters(
    n_samples: int = 600,
    contamination: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    2D Multimodal normal modes (Day Operation vs Night Shift) with scattered outliers.
    """
    np.random.seed(seed)
    n_anomalies = max(4, int(n_samples * contamination))
    n_normal = n_samples - n_anomalies
    n1 = n_normal // 2
    n2 = n_normal - n1

    c1 = np.random.normal(loc=[-2.5, -2.0], scale=0.7, size=(n1, 2))
    c2 = np.random.normal(loc=[2.5, 2.0], scale=0.8, size=(n2, 2))
    pts_norm = np.vstack([c1, c2])

    # Outliers scattered in perimeter / void between clusters
    pts_anom = np.random.uniform(low=[-5.5, -5.5], high=[5.5, 5.5], size=(n_anomalies, 2))

    df_norm = pd.DataFrame(pts_norm, columns=["Feature_X1", "Feature_X2"]).assign(is_anomaly=0)
    df_anom = pd.DataFrame(pts_anom, columns=["Feature_X1", "Feature_X2"]).assign(is_anomaly=1)

    df = pd.concat([df_norm, df_anom], ignore_index=True)
    df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    return df, ["Feature_X1", "Feature_X2"], "is_anomaly"


def generate_masking_demonstration(
    n_samples: int = 500,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Dataset demonstrating the Masking & Swamping phenomena.
    Standard Z-Score gets blinded by extreme clustered outliers that inflate mean and variance,
    while Modified Z-Score (MAD) remains impervious.
    """
    np.random.seed(seed)
    normal = np.random.normal(loc=50.0, scale=4.0, size=n_samples - 8)
    subtle_anomalies = [64.0, 65.5, 66.0]  # Real anomalies
    extreme_maskers = [120.0, 125.0, 130.0, 135.0, 140.0]  # Outliers that inflate std

    data = np.concatenate([normal, subtle_anomalies, extreme_maskers])
    labels = np.zeros(len(data), dtype=int)
    labels[len(normal):] = 1

    df = pd.DataFrame({
        "metric_value": data,
        "is_anomaly": labels
    })
    return df, ["metric_value"], "is_anomaly"


# =============================================================================
# 3. STREAMING TIME-SERIES TELEMETRY GENERATOR
# =============================================================================

def generate_streaming_telemetry(
    n_steps: int = 400,
    noise_level: float = 1.0,
    seed: int = 42
) -> pd.DataFrame:
    """
    Time series streaming signal with:
    - Diurnal cyclical wave (sinusoid)
    - Slow linear drift
    - Injected point spikes (sudden transient leaps)
    - Contextual anomalies (high value during normally quiet off-peak trough)
    - Persistent level-shift (step function change)
    """
    np.random.seed(seed)
    time_idx = np.arange(n_steps)

    # Base seasonal cycle + baseline drift
    cycle = 15.0 * np.sin(2 * np.pi * time_idx / 50.0)
    drift = 0.04 * time_idx
    noise = np.random.normal(loc=0.0, scale=noise_level, size=n_steps)
    signal = 60.0 + cycle + drift + noise

    labels = np.zeros(n_steps, dtype=int)

    # 1. Point Spikes
    spikes = [45, 110, 185, 290]
    for s_idx in spikes:
        if s_idx < n_steps:
            signal[s_idx] += np.random.choice([25.0, -22.0])
            labels[s_idx] = 1

    # 2. Level shift from step 320 to 350
    if n_steps > 350:
        signal[320:350] += 18.0
        labels[320:350] = 1

    # 3. Contextual anomaly at a trough
    trough_step = 75
    if trough_step < n_steps:
        signal[trough_step] = 85.0
        labels[trough_step] = 1

    df = pd.DataFrame({
        "timestamp_step": time_idx,
        "metric_stream": signal,
        "is_anomaly": labels
    })
    return df


# =============================================================================
# 4. BENCHMARK LOADER & ARCHETYPE PRESETS
# =============================================================================

def load_benchmark_dataset(
    dataset_name: str,
    contamination: float = 0.05,
    seed: int = 42
) -> Tuple[pd.DataFrame, List[str], str]:
    """
    Universal dataset loader for both real-world domain telemetry and 2D manifolds.
    """
    ds_clean = dataset_name.lower()

    if "fraud" in ds_clean or "credit" in ds_clean:
        return generate_financial_fraud_telemetry(contamination=contamination, seed=seed)
    elif "cloud" in ds_clean or "server" in ds_clean:
        return generate_cloud_server_telemetry(contamination=contamination, seed=seed)
    elif "iot" in ds_clean or "sensor" in ds_clean:
        return generate_iot_sensor_telemetry(contamination=contamination, seed=seed)
    elif "vital" in ds_clean or "medical" in ds_clean:
        return generate_medical_vital_signs(contamination=contamination, seed=seed)
    elif "correlated" in ds_clean or "covariance" in ds_clean:
        return generate_correlated_gaussian(contamination=contamination, seed=seed)
    elif "cluster" in ds_clean or "multimodal" in ds_clean:
        return generate_multimodal_clusters(contamination=contamination, seed=seed)
    elif "masking" in ds_clean:
        return generate_masking_demonstration(seed=seed)
    else:
        return generate_financial_fraud_telemetry(contamination=contamination, seed=seed)


def get_archetype_presets() -> Dict[str, Dict[str, Any]]:
    """
    Preconfigured scenario archetypes for quick experimentation.
    """
    return {
        "Financial Payment Fraud Burst": {
            "dataset": "Financial Transactions & Fraud",
            "method": "ensemble",
            "z_threshold": 3.0,
            "mod_z_threshold": 3.5,
            "iqr_k": 1.5,
            "mahalanobis_alpha": 0.01,
            "description": "Multi-variate transaction anomalies with rapid velocity card-testing and stolen card overseas cashouts."
        },
        "Cloud Server Deadlock & Resource Peg": {
            "dataset": "Cloud Server Infrastructure Telemetry",
            "method": "mahalanobis",
            "z_threshold": 2.8,
            "mod_z_threshold": 3.2,
            "iqr_k": 1.5,
            "mahalanobis_alpha": 0.005,
            "description": "Correlated memory leaks, thread starvation, and 5xx error cascade detection."
        },
        "Industrial Turbine Bearing Seizure": {
            "dataset": "Industrial IoT Sensor Telemetry",
            "method": "modified_z_score",
            "z_threshold": 3.0,
            "mod_z_threshold": 3.5,
            "iqr_k": 1.5,
            "mahalanobis_alpha": 0.01,
            "description": "Robust MAD detection against heavy-tailed machine vibration and temperature surges."
        },
        "Off-Diagonal Covariance Breach (2D)": {
            "dataset": "2D Correlated Gaussian (Covariance Breach)",
            "method": "mahalanobis",
            "z_threshold": 3.0,
            "mod_z_threshold": 3.5,
            "iqr_k": 1.5,
            "mahalanobis_alpha": 0.01,
            "description": "Demonstrates points looking normal in 1D projections but violently violating multivariate correlation."
        }
    }
