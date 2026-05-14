"""Estimate refined-oil model parameters from the calculation document.

This script is intentionally self-contained. It does not import the existing
model scripts in this repository. The implementation follows the supplied
calculation method document, with one practical adjustment: exchange rate is
used directly as an observed regressor instead of being approximated by segment
constants. This keeps the regression linear while avoiding extra approximation.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd


def find_project_root(start_file: Path) -> Path:
    for candidate in start_file.resolve().parents:
        if (candidate / "data").exists() and (candidate / "result").exists():
            return candidate
    raise RuntimeError("Could not find project root containing data/ and result/.")


ROOT = find_project_root(Path(__file__))
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "result"


BARREL_PER_TON = 7.33
OIL_FLOOR = 40.0
LAMBDA_FACTOR_ABOVE_130 = 0.30
RULE_THRESHOLD = 50.0
EPS = 1e-9
MIN_SEGMENT_ROWS = 6


FIELD_MAP = {
    "domestic_file": "result/domastic_price.csv",
    "rare_file": "data/rare-domastic.csv",
    "wti_file": "data/wti-daily.csv",
    "brent_file": "data/brent-daily.csv",
    "basket_file": "data/basket-daily.csv",
    "exchange_file": "data/cny_usd_exchange_rate.csv",
    "date": "domastic_price.date",
    "notice_date": "rare-domastic.notice_date, fallback to date",
    "WTI": "wti-daily.price",
    "Brent": "brent-daily.price",
    "Basket": "basket-daily.price",
    "exchange_rate": "cny_usd_exchange_rate second column",
    "gasoline_price": "domastic_price.gasoline_price_after",
    "diesel_price": "domastic_price.diesel_price_after",
    "gasoline_delta": "domastic_price.gasoline_change",
    "diesel_delta": "domastic_price.diesel_change",
    "is_special_regulated": "domastic_price.is_special_regulated",
}


@dataclass(frozen=True)
class FuelConfig:
    name: str
    price_col: str
    delta_col: str
    prev_price_col: str


FUELS = {
    "gasoline": FuelConfig(
        name="gasoline",
        price_col="gasoline_price_after",
        delta_col="gasoline_change",
        prev_price_col="prev_gasoline_price",
    ),
    "diesel": FuelConfig(
        name="diesel",
        price_col="diesel_price_after",
        delta_col="diesel_change",
        prev_price_col="prev_diesel_price",
    ),
}


def parse_bool_series(series: pd.Series) -> pd.Series:
    return (
        series.astype(str)
        .str.strip()
        .str.upper()
        .isin(["TRUE", "1", "YES", "Y"])
    )


def read_domestic_data() -> pd.DataFrame:
    domestic_path = RESULT_DIR / "domastic_price.csv"
    domestic = pd.read_csv(domestic_path)
    domestic["date"] = pd.to_datetime(domestic["date"], errors="coerce")
    domestic["is_changed"] = parse_bool_series(domestic["is_changed"])
    domestic["is_special_regulated"] = parse_bool_series(
        domestic["is_special_regulated"]
    )

    numeric_cols = [
        "gasoline_change",
        "diesel_change",
        "gasoline_price_after",
        "diesel_price_after",
    ]
    for col in numeric_cols:
        domestic[col] = pd.to_numeric(domestic[col], errors="coerce")

    rare_path = DATA_DIR / "rare-domastic.csv"
    if rare_path.exists():
        rare = pd.read_csv(rare_path)
        rare["date"] = pd.to_datetime(rare["date"], errors="coerce")
        rare["notice_date"] = pd.to_datetime(rare["notice_date"], errors="coerce")
        rare = rare[["date", "notice_date", "notice_title", "source_url"]]
        domestic = domestic.merge(rare, on="date", how="left")
    else:
        domestic["notice_date"] = pd.NaT
        domestic["notice_title"] = ""
        domestic["source_url"] = ""

    domestic["notice_date"] = domestic["notice_date"].fillna(domestic["date"])
    domestic = domestic.dropna(subset=["date", *numeric_cols])
    domestic = domestic.sort_values("date").reset_index(drop=True)

    for fuel in FUELS.values():
        domestic[fuel.prev_price_col] = domestic[fuel.price_col].shift(1)
        first_mask = domestic[fuel.prev_price_col].isna()
        domestic.loc[first_mask, fuel.prev_price_col] = (
            domestic.loc[first_mask, fuel.price_col]
            - domestic.loc[first_mask, fuel.delta_col]
        )

    return domestic


def read_daily_price(file_name: str) -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / file_name)
    data = data.iloc[:, 0:2]
    data.columns = ["date", "price"]
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["price"] = pd.to_numeric(data["price"], errors="coerce")
    data = data.dropna(subset=["date", "price"])
    return data.sort_values("date").reset_index(drop=True)


def read_exchange_rate() -> pd.DataFrame:
    data = pd.read_csv(DATA_DIR / "cny_usd_exchange_rate.csv")
    data = data.iloc[:, 0:2]
    data.columns = ["exchange_date", "exchange_rate"]
    data["exchange_date"] = pd.to_datetime(data["exchange_date"], errors="coerce")
    data["exchange_rate"] = pd.to_numeric(data["exchange_rate"], errors="coerce")
    data = data.dropna(subset=["exchange_date", "exchange_rate"])
    return data.sort_values("exchange_date").reset_index(drop=True)


def latest_window_means(
    anchor_dates: Iterable[pd.Timestamp],
    oil_df: pd.DataFrame,
    oil_name: str,
    window_size: int,
    include_anchor: bool,
) -> pd.DataFrame:
    dates = oil_df["date"].to_numpy(dtype="datetime64[ns]")
    prices = oil_df["price"].to_numpy(dtype=float)
    side = "right" if include_anchor else "left"
    rows = []

    for anchor in anchor_dates:
        anchor64 = np.datetime64(pd.Timestamp(anchor).to_datetime64())
        end = int(np.searchsorted(dates, anchor64, side=side))
        start = max(0, end - window_size)

        if end <= start:
            rows.append(
                {
                    f"{oil_name}_mean": np.nan,
                    f"{oil_name}_valid_days": 0,
                    f"{oil_name}_window_start": pd.NaT,
                    f"{oil_name}_window_end": pd.NaT,
                }
            )
            continue

        window_prices = prices[start:end]
        window_dates = dates[start:end]
        rows.append(
            {
                f"{oil_name}_mean": float(np.mean(window_prices)),
                f"{oil_name}_valid_days": int(len(window_prices)),
                f"{oil_name}_window_start": pd.Timestamp(window_dates[0]),
                f"{oil_name}_window_end": pd.Timestamp(window_dates[-1]),
            }
        )

    return pd.DataFrame(rows)


def load_model_data(window_size: int, include_anchor: bool) -> pd.DataFrame:
    domestic = read_domestic_data()
    domestic["anchor_date"] = domestic["notice_date"]

    for oil_name, file_name in [
        ("wti", "wti-daily.csv"),
        ("brent", "brent-daily.csv"),
        ("basket", "basket-daily.csv"),
    ]:
        oil_df = read_daily_price(file_name)
        average = latest_window_means(
            anchor_dates=domestic["anchor_date"],
            oil_df=oil_df,
            oil_name=oil_name,
            window_size=window_size,
            include_anchor=include_anchor,
        )
        domestic = pd.concat([domestic, average], axis=1)

    exchange = read_exchange_rate()
    data = pd.merge_asof(
        domestic.sort_values("anchor_date"),
        exchange,
        left_on="anchor_date",
        right_on="exchange_date",
        direction="backward",
    ).sort_values("date")

    required = [
        "wti_mean",
        "brent_mean",
        "basket_mean",
        "exchange_rate",
        "gasoline_price_after",
        "diesel_price_after",
    ]
    data = data.dropna(subset=required).reset_index(drop=True)
    return data


def build_weight_grid(step: float) -> List[Tuple[float, float, float]]:
    scale = int(round(1.0 / step))
    if not math.isclose(scale * step, 1.0, abs_tol=1e-9):
        raise ValueError("step must divide 1 exactly, for example 0.02 or 0.01")

    weights = []
    for i in range(scale + 1):
        for j in range(scale + 1 - i):
            w1 = i / scale
            w2 = j / scale
            w3 = 1.0 - w1 - w2
            weights.append((round(w1, 6), round(w2, 6), round(w3, 6)))
    return weights


def weighted_oil(data: pd.DataFrame, weights: Tuple[float, float, float]) -> np.ndarray:
    w1, w2, w3 = weights
    return (
        w1 * data["wti_mean"].to_numpy(dtype=float)
        + w2 * data["brent_mean"].to_numpy(dtype=float)
        + w3 * data["basket_mean"].to_numpy(dtype=float)
    )


def filtered_oil(weighted: np.ndarray) -> np.ndarray:
    return np.maximum(OIL_FLOOR, weighted)


def profit_factor(oil: np.ndarray) -> np.ndarray:
    oil = np.asarray(oil, dtype=float)
    return np.where(oil <= 80.0, 1.0, np.where(oil <= 130.0, (130.0 - oil) / 50.0, 0.0))


def oil_transfer_input(oil: np.ndarray, exchange_rate: np.ndarray) -> np.ndarray:
    """Known oil-cost regressor, including the >130 USD low-pass-through rule."""
    oil = np.asarray(oil, dtype=float)
    effective_oil = np.where(
        oil <= 130.0,
        oil,
        130.0 + LAMBDA_FACTOR_ABOVE_130 * (oil - 130.0),
    )
    return effective_oil * exchange_rate * BARREL_PER_TON


def design_matrix(data: pd.DataFrame, weights: Tuple[float, float, float]) -> np.ndarray:
    oil = filtered_oil(weighted_oil(data, weights))
    exchange = data["exchange_rate"].to_numpy(dtype=float)
    x_oil = oil_transfer_input(oil, exchange)
    h_oil = profit_factor(oil)
    return np.column_stack([x_oil, h_oil, np.ones(len(data))])


def fit_linear_params(x_matrix: np.ndarray, y_vector: np.ndarray) -> np.ndarray:
    if len(y_vector) < x_matrix.shape[1]:
        raise ValueError("not enough rows for regression")

    try:
        from scipy.optimize import lsq_linear

        lower = np.array([0.0, 0.0, -np.inf])
        upper = np.array([np.inf, np.inf, np.inf])
        result = lsq_linear(x_matrix, y_vector, bounds=(lower, upper), lsmr_tol="auto")
        return result.x.astype(float)
    except Exception:
        coef, *_ = np.linalg.lstsq(x_matrix, y_vector, rcond=None)
        return coef.astype(float)


def predict_price(
    data: pd.DataFrame,
    weights: Tuple[float, float, float],
    alpha: float,
    pi0: float,
    beta: float,
) -> np.ndarray:
    x_matrix = design_matrix(data, weights)
    return x_matrix @ np.array([alpha, pi0, beta], dtype=float)


def metric_block(error: np.ndarray, actual: np.ndarray | None = None) -> Dict[str, float]:
    error = np.asarray(error, dtype=float)
    finite = np.isfinite(error)
    if not finite.any():
        return {"mae": np.nan, "rmse": np.nan, "mape": np.nan}

    err = error[finite]
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err**2)))
    mape = np.nan
    if actual is not None:
        actual_arr = np.asarray(actual, dtype=float)[finite]
        mask = np.abs(actual_arr) > EPS
        if mask.any():
            mape = float(np.mean(np.abs(err[mask] / actual_arr[mask])))
    return {"mae": mae, "rmse": rmse, "mape": mape}


def build_clean_mask(data: pd.DataFrame, fuel: FuelConfig) -> pd.Series:
    prev_special = data["is_special_regulated"].shift(1).fillna(True)
    prev_delta = data[fuel.delta_col].shift(1)
    return (
        (~data["is_special_regulated"])
        & (~prev_special)
        & (prev_delta.abs() >= RULE_THRESHOLD)
        & (data[fuel.delta_col].abs() >= RULE_THRESHOLD)
    )


def prepare_prediction_frame(
    data: pd.DataFrame,
    fuel: FuelConfig,
    weights: Tuple[float, float, float],
    alpha: float,
    pi0: float,
    beta: float,
    mask: pd.Series,
) -> pd.DataFrame:
    frame = data.loc[mask].copy()
    pred_price = predict_price(frame, weights, alpha, pi0, beta)
    actual_price = frame[fuel.price_col].to_numpy(dtype=float)
    prev_actual = frame[fuel.prev_price_col].to_numpy(dtype=float)
    actual_delta = actual_price - prev_actual
    pred_delta = pred_price - prev_actual

    return pd.DataFrame(
        {
            "date": frame["date"].dt.strftime("%Y-%m-%d"),
            "actual_price": actual_price,
            "pred_price": pred_price,
            "error": pred_price - actual_price,
            "abs_error": np.abs(pred_price - actual_price),
            "actual_delta": actual_delta,
            "pred_delta": pred_delta,
            "delta_error": pred_delta - actual_delta,
            "is_special_regulated": frame["is_special_regulated"].astype(bool).to_numpy(),
        }
    )


def fit_segment_parameters(
    clean_data: pd.DataFrame,
    fuel: FuelConfig,
    weights: Tuple[float, float, float],
) -> pd.DataFrame:
    rows = []
    clean_data = clean_data.copy()
    clean_data["segment"] = clean_data["date"].dt.year.astype(str)

    for segment, seg_df in clean_data.groupby("segment"):
        y = seg_df[fuel.price_col].to_numpy(dtype=float)
        sample_count = int(len(seg_df))
        row = {
            "segment": segment,
            "sample_count": sample_count,
            "mu_bar": float(seg_df["exchange_rate"].mean()),
            "alpha_s": np.nan,
            "beta_s": np.nan,
            "pi0_s": np.nan,
            "regression_mae": np.nan,
            "valid": False,
            "invalid_reason": "",
        }
        if sample_count < MIN_SEGMENT_ROWS:
            row["invalid_reason"] = f"sample_count_less_than_{MIN_SEGMENT_ROWS}"
            rows.append(row)
            continue

        x_matrix = design_matrix(seg_df, weights)
        try:
            alpha_s, pi0_s, beta_s = fit_linear_params(x_matrix, y)
            pred = x_matrix @ np.array([alpha_s, pi0_s, beta_s], dtype=float)
            row.update(
                {
                    "alpha_s": float(alpha_s),
                    "beta_s": float(beta_s),
                    "pi0_s": float(pi0_s),
                    "regression_mae": float(np.mean(np.abs(pred - y))),
                    "valid": True,
                }
            )
        except Exception as exc:
            row["invalid_reason"] = str(exc)
        rows.append(row)

    return pd.DataFrame(rows)


def estimate_for_fuel(
    data: pd.DataFrame,
    fuel: FuelConfig,
    weight_grid: List[Tuple[float, float, float]],
) -> Dict[str, object]:
    clean_mask = build_clean_mask(data, fuel)
    clean = data.loc[clean_mask].copy()
    non_special_mask = ~data["is_special_regulated"]
    non_special = data.loc[non_special_mask].copy()

    if len(clean) < 10:
        raise ValueError(f"not enough clean rows for {fuel.name}: {len(clean)}")

    y_clean = clean[fuel.price_col].to_numpy(dtype=float)
    rows = []

    for weights in weight_grid:
        x_clean = design_matrix(clean, weights)
        alpha, pi0, beta = fit_linear_params(x_clean, y_clean)

        clean_pred = x_clean @ np.array([alpha, pi0, beta], dtype=float)
        regression_mae = float(np.mean(np.abs(clean_pred - y_clean)))

        pred_frame = prepare_prediction_frame(
            data=data,
            fuel=fuel,
            weights=weights,
            alpha=alpha,
            pi0=pi0,
            beta=beta,
            mask=non_special_mask,
        )
        price_metrics = metric_block(
            pred_frame["error"].to_numpy(dtype=float),
            pred_frame["actual_price"].to_numpy(dtype=float),
        )
        volatility_metrics = metric_block(
            pred_frame["delta_error"].to_numpy(dtype=float),
            pred_frame["actual_delta"].to_numpy(dtype=float),
        )

        rows.append(
            {
                "w1": weights[0],
                "w2": weights[1],
                "w3": weights[2],
                "alpha": float(alpha),
                "beta": float(beta),
                "pi0": float(pi0),
                "regression_mae": regression_mae,
                "full_model_mae": price_metrics["mae"],
                "full_model_rmse": price_metrics["rmse"],
                "full_model_mape": price_metrics["mape"],
                "volatility_mae": volatility_metrics["mae"],
                "volatility_rmse": volatility_metrics["rmse"],
                "volatility_mape": volatility_metrics["mape"],
                "sample_count": int(len(clean)),
                "validation_count": int(len(non_special)),
            }
        )

    grid = pd.DataFrame(rows)
    grid = grid.sort_values(
        ["full_model_mae", "full_model_rmse", "regression_mae"]
    ).reset_index(drop=True)
    grid["price_rank"] = np.arange(1, len(grid) + 1)
    grid["volatility_rank"] = (
        grid["volatility_mae"].rank(method="first", ascending=True).astype(int)
    )

    best_price = grid.iloc[0].copy()
    best_volatility = grid.sort_values(
        ["volatility_mae", "volatility_rmse", "regression_mae"]
    ).iloc[0].copy()

    price_weights = (
        float(best_price["w1"]),
        float(best_price["w2"]),
        float(best_price["w3"]),
    )
    full_prediction = prepare_prediction_frame(
        data=data,
        fuel=fuel,
        weights=price_weights,
        alpha=float(best_price["alpha"]),
        pi0=float(best_price["pi0"]),
        beta=float(best_price["beta"]),
        mask=non_special_mask,
    )

    segment_params = fit_segment_parameters(clean, fuel, price_weights)

    return {
        "grid": grid,
        "best_price": best_price,
        "best_volatility": best_volatility,
        "full_prediction": full_prediction,
        "segment_params": segment_params,
        "clean_count": int(len(clean)),
        "non_special_count": int(len(non_special)),
    }


def best_row_to_output(row: pd.Series, fuel_name: str, selection_metric: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                f"w1_{fuel_name}": row["w1"],
                f"w2_{fuel_name}": row["w2"],
                f"w3_{fuel_name}": row["w3"],
                f"alpha_{fuel_name}": row["alpha"],
                f"beta_{fuel_name}": row["beta"],
                f"pi0_{fuel_name}": row["pi0"],
                "selection_metric": selection_metric,
                "full_model_mae": row["full_model_mae"],
                "full_model_rmse": row["full_model_rmse"],
                "full_model_mape": row["full_model_mape"],
                "volatility_mae": row["volatility_mae"],
                "volatility_rmse": row["volatility_rmse"],
                "volatility_mape": row["volatility_mape"],
                "sample_count": row["sample_count"],
                "validation_count": row["validation_count"],
            }
        ]
    )


def compute_special_mu(
    data: pd.DataFrame,
    fuel: FuelConfig,
    best_row: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    weights = (float(best_row["w1"]), float(best_row["w2"]), float(best_row["w3"]))
    special = data.loc[data["is_special_regulated"]].copy()
    if special.empty:
        detail = pd.DataFrame(
            columns=[
                "date",
                "prev_actual_price",
                "actual_price",
                "P_rule",
                "actual_delta",
                "rule_delta",
                "mu_i",
                "valid",
                "invalid_reason",
            ]
        )
        summary = summarize_mu(detail)
        return detail, summary

    p_rule = predict_price(
        special,
        weights,
        alpha=float(best_row["alpha"]),
        pi0=float(best_row["pi0"]),
        beta=float(best_row["beta"]),
    )
    prev_actual = special[fuel.prev_price_col].to_numpy(dtype=float)
    actual_price = special[fuel.price_col].to_numpy(dtype=float)
    actual_delta = actual_price - prev_actual
    rule_delta = p_rule - prev_actual

    rows = []
    for i, (_, row) in enumerate(special.iterrows()):
        invalid_reason = ""
        valid = True
        mu_i = np.nan
        if not np.isfinite(rule_delta[i]) or abs(rule_delta[i]) <= EPS:
            valid = False
            invalid_reason = "rule_delta_close_to_zero"
        elif abs(rule_delta[i]) < RULE_THRESHOLD:
            valid = False
            invalid_reason = "rule_delta_below_threshold"
        else:
            mu_i = float(actual_delta[i] / rule_delta[i])

        rows.append(
            {
                "date": row["date"].strftime("%Y-%m-%d"),
                "prev_actual_price": float(prev_actual[i]),
                "actual_price": float(actual_price[i]),
                "P_rule": float(p_rule[i]),
                "actual_delta": float(actual_delta[i]),
                "rule_delta": float(rule_delta[i]),
                "mu_i": mu_i,
                "valid": valid,
                "invalid_reason": invalid_reason,
            }
        )

    detail = pd.DataFrame(rows)
    summary = summarize_mu(detail)
    return detail, summary


def summarize_mu(detail: pd.DataFrame) -> pd.DataFrame:
    valid = detail[detail.get("valid", False) == True].copy()
    valid = valid[np.isfinite(valid["mu_i"])] if "mu_i" in valid else valid

    if valid.empty:
        return pd.DataFrame(
            [
                {
                    "mu_mean": np.nan,
                    "mu_std": np.nan,
                    "mu_cv": np.nan,
                    "valid_count": 0,
                    "can_be_constant": "insufficient_valid_mu",
                }
            ]
        )

    mu = valid["mu_i"].to_numpy(dtype=float)
    mu_mean = float(np.mean(mu))
    mu_std = float(np.std(mu, ddof=0))
    mu_cv = np.nan if abs(mu_mean) <= EPS else float(abs(mu_std / mu_mean))

    if not np.isfinite(mu_cv):
        can_be_constant = "not_recommended"
    elif mu_cv < 0.05:
        can_be_constant = "highly_stable_constant"
    elif mu_cv < 0.10:
        can_be_constant = "basically_stable_constant"
    else:
        can_be_constant = "not_recommended"

    return pd.DataFrame(
        [
            {
                "mu_mean": mu_mean,
                "mu_std": mu_std,
                "mu_cv": mu_cv,
                "valid_count": int(len(mu)),
                "can_be_constant": can_be_constant,
            }
        ]
    )


def fmt(value: object, digits: int = 6) -> str:
    try:
        value_float = float(value)
    except Exception:
        return str(value)
    if not np.isfinite(value_float):
        return "NA"
    return f"{value_float:.{digits}f}"


def write_report(
    data: pd.DataFrame,
    results: Dict[str, Dict[str, object]],
    special_summaries: Dict[str, pd.DataFrame],
    output_path: Path,
    window_size: int,
    include_anchor: bool,
) -> None:
    gas_best = results["gasoline"]["best_price"]
    diesel_best = results["diesel"]["best_price"]
    gas_vol = results["gasoline"]["best_volatility"]
    diesel_vol = results["diesel"]["best_volatility"]
    gas_mu = special_summaries["gasoline"].iloc[0]
    diesel_mu = special_summaries["diesel"].iloc[0]

    def fuel_block(label: str, row: pd.Series, vol_row: pd.Series, mu_row: pd.Series) -> str:
        return f"""## {label}模型结果

- 价格损失最优 w1, w2, w3: {fmt(row['w1'])}, {fmt(row['w2'])}, {fmt(row['w3'])}
- alpha: {fmt(row['alpha'])}
- beta: {fmt(row['beta'])}
- pi0: {fmt(row['pi0'])}
- 完整静态模型价格 MAE/RMSE/MAPE: {fmt(row['full_model_mae'], 4)}, {fmt(row['full_model_rmse'], 4)}, {fmt(row['full_model_mape'], 6)}
- 价格波动损失最优 w1, w2, w3: {fmt(vol_row['w1'])}, {fmt(vol_row['w2'])}, {fmt(vol_row['w3'])}
- 价格波动 MAE/RMSE/MAPE: {fmt(vol_row['volatility_mae'], 4)}, {fmt(vol_row['volatility_rmse'], 4)}, {fmt(vol_row['volatility_mape'], 6)}
- 特殊调控 mu_i 均值/标准差/CV/有效样本数: {fmt(mu_row['mu_mean'])}, {fmt(mu_row['mu_std'])}, {fmt(mu_row['mu_cv'])}, {int(mu_row['valid_count'])}
- mu_i 是否近似常数: {mu_row['can_be_constant']}
"""

    diff_lines = [
        f"- w1 差异: {fmt(abs(float(gas_best['w1']) - float(diesel_best['w1'])))}",
        f"- w2 差异: {fmt(abs(float(gas_best['w2']) - float(diesel_best['w2'])))}",
        f"- w3 差异: {fmt(abs(float(gas_best['w3']) - float(diesel_best['w3'])))}",
        f"- alpha 差异: {fmt(abs(float(gas_best['alpha']) - float(diesel_best['alpha'])))}",
        f"- beta 差异: {fmt(abs(float(gas_best['beta']) - float(diesel_best['beta'])))}",
        f"- pi0 差异: {fmt(abs(float(gas_best['pi0']) - float(diesel_best['pi0'])))}",
        f"- 特殊调控 mu 均值差异: {fmt(abs(float(gas_mu['mu_mean']) - float(diesel_mu['mu_mean'])))}",
    ]

    report = f"""# 参数估计报告

## 一、数据来源与字段识别结果

- 国内调价数据: {FIELD_MAP['domestic_file']}
- 特殊公告补充数据: {FIELD_MAP['rare_file']}
- WTI 数据: {FIELD_MAP['wti_file']}
- Brent 数据: {FIELD_MAP['brent_file']}
- Basket 数据: {FIELD_MAP['basket_file']}
- 汇率数据: {FIELD_MAP['exchange_file']}
- 日期字段: {FIELD_MAP['date']}
- 公告日字段: {FIELD_MAP['notice_date']}
- WTI 字段: {FIELD_MAP['WTI']}
- Brent 字段: {FIELD_MAP['Brent']}
- Basket 字段: {FIELD_MAP['Basket']}
- 汇率字段: {FIELD_MAP['exchange_rate']}
- 汽油价格字段: {FIELD_MAP['gasoline_price']}
- 柴油价格字段: {FIELD_MAP['diesel_price']}
- 汽油调价幅度字段: {FIELD_MAP['gasoline_delta']}
- 柴油调价幅度字段: {FIELD_MAP['diesel_delta']}
- 特殊调控标记字段: {FIELD_MAP['is_special_regulated']}

数据行数: {len(data)}。特殊调控样本数: {int(data['is_special_regulated'].sum())}。非特殊调控样本数: {int((~data['is_special_regulated']).sum())}。

国际油价按公告日前最近 {window_size} 个可得交易日取均值，include_anchor={include_anchor}。汇率按公告日前最近可得日汇率合并。回归时直接使用每日汇率作为已知自变量，因此模型仍为线性回归，不再额外做汇率分段常数近似；年度 segment 文件仅作为诊断输出。

{fuel_block('二、汽油', gas_best, gas_vol, gas_mu)}

{fuel_block('三、柴油', diesel_best, diesel_vol, diesel_mu)}

## 四、汽油和柴油结果对比

{chr(10).join(diff_lines)}

若汽油和柴油参数存在明显差异，可能来自两类油品税费、品质差异、最高零售价基准不同，以及调价幅度四舍五入或地方口径差异。

## 五、误差较大时的可能原因

- 调价窗口可能需要改用 10 个自然日、10 个工作日或不含公告日窗口做稳健性比较。
- 国内调价日期可能存在公告日与生效日差异。
- 特殊调控标记可能仍有漏标或误标。
- 40、80、130 美元分段利润函数可能与实际政策口径存在细节差异。
- 汇率使用公告日最近值，若真实采用窗口均值，误差会扩大。
- 干净样本筛选后部分年份样本较少，年度局部参数不稳定。
- 未调价样本的实际价格包含门槛和累计影响，而静态完整模型不进行 carry 和 50 元门槛递推。
- 特殊调控样本较少，mu_i 的稳定性结论只能作为经验判断。
"""
    output_path.write_text(report, encoding="utf-8-sig")


def write_outputs(
    results: Dict[str, Dict[str, object]],
    special_details: Dict[str, pd.DataFrame],
    special_summaries: Dict[str, pd.DataFrame],
) -> None:
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for fuel_name, result in results.items():
        grid = result["grid"]
        best_price = result["best_price"]
        best_volatility = result["best_volatility"]

        grid.to_csv(
            RESULT_DIR / f"grid_search_parameters_{fuel_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        best_row_to_output(best_price, fuel_name, "price_mae").to_csv(
            RESULT_DIR / f"best_parameters_{fuel_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        best_row_to_output(best_volatility, fuel_name, "volatility_mae").to_csv(
            RESULT_DIR / f"best_parameters_{fuel_name}_volatility.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["segment_params"].to_csv(
            RESULT_DIR / f"segment_parameters_{fuel_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        result["full_prediction"].to_csv(
            RESULT_DIR / f"full_model_prediction_{fuel_name}_non_special.csv",
            index=False,
            encoding="utf-8-sig",
        )
        special_details[fuel_name].to_csv(
            RESULT_DIR / f"special_mu_{fuel_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )
        special_summaries[fuel_name].to_csv(
            RESULT_DIR / f"special_mu_summary_{fuel_name}.csv",
            index=False,
            encoding="utf-8-sig",
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate parameters from DOCX method")
    parser.add_argument("--grid-step", type=float, default=0.02)
    parser.add_argument("--window-size", type=int, default=10)
    parser.add_argument("--exclude-anchor", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    include_anchor = not args.exclude_anchor
    data = load_model_data(
        window_size=args.window_size,
        include_anchor=include_anchor,
    )
    weight_grid = build_weight_grid(args.grid_step)

    results: Dict[str, Dict[str, object]] = {}
    special_details: Dict[str, pd.DataFrame] = {}
    special_summaries: Dict[str, pd.DataFrame] = {}

    for fuel_name, fuel in FUELS.items():
        results[fuel_name] = estimate_for_fuel(data, fuel, weight_grid)
        detail, summary = compute_special_mu(
            data=data,
            fuel=fuel,
            best_row=results[fuel_name]["best_price"],
        )
        special_details[fuel_name] = detail
        special_summaries[fuel_name] = summary

    write_outputs(results, special_details, special_summaries)
    write_report(
        data=data,
        results=results,
        special_summaries=special_summaries,
        output_path=RESULT_DIR / "parameter_estimation_report.md",
        window_size=args.window_size,
        include_anchor=include_anchor,
    )

    for fuel_name, result in results.items():
        best = result["best_price"]
        print(
            f"{fuel_name}: w=({best['w1']:.2f}, {best['w2']:.2f}, {best['w3']:.2f}), "
            f"alpha={best['alpha']:.6f}, beta={best['beta']:.4f}, "
            f"pi0={best['pi0']:.4f}, MAE={best['full_model_mae']:.4f}"
        )
    print(f"outputs written to {RESULT_DIR}")


if __name__ == "__main__":
    main()
