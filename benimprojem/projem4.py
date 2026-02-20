# ============================================================
# SKA 7 (Temiz Enerji) - TEK ALGORİTMA: ARIMA (+ Residual Diagnostics)
# Yönerge uyumu:
# - Zaman serisi: ARIMA (Seçilen tek algoritma)
# Ek geliştirmeler (Rapor için güçlü görseller):
# - Residual (artık) analiz grafikleri: residual time plot, histogram, ACF, QQ plot
# - Ljung-Box testi (opsiyonel) -> artıklar beyaz gürültü mü?
#
# Üretilen görseller (outputs/):
# 01_missing_map.png
# 02_raw_vs_clean.png
# 03_iqr_outliers.png
# 04_train_test_split.png
# 05_test_forecast.png
# 06_forecast_with_ci.png
# 07_residuals_time.png
# 08_residual_hist.png
# 09_residual_acf.png
# 10_residual_qq.png
# 11_ljungbox.csv (opsiyonel)
# ============================================================

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.stats.diagnostic import acorr_ljungbox
import statsmodels.api as sm


# =========================
# 0) AYARLAR
# =========================
DATA_FILE = r"C:\Users\enesd\Desktop\bizimproje\clean_fuels.csv"
COUNTRY_CODE = "TUR"

YEAR_MIN = 1960
YEAR_MAX = 2024

# Veri seçme modu:
# - "contiguous": NaN olmayan en uzun kesintisiz blok
# - "last_n": son N yıl
SELECT_MODE = "contiguous"
LAST_N_YEARS = 25

MIN_TEST_SIZE = 3
FORECAST_YEARS = list(range(2023, 2031))  # 2023–2030

USE_NAIVE_BASELINE = False  # "tek algoritma" kuralı çok katıysa False
OUTPUT_DIR = "outputs"      # görseller buraya kaydedilecek
SAVE_FIGS = True            # rapor için True bırak

# ARIMA arama sınırları
P_MAX, D_MAX, Q_MAX = 3, 2, 3

# Ljung-Box opsiyonel
RUN_LJUNG_BOX = True


# =========================
# yardımcı: klasör + fig kaydet
# =========================
def _ensure_outdir(path: str):
    os.makedirs(path, exist_ok=True)

def savefig(name: str):
    if SAVE_FIGS:
        _ensure_outdir(OUTPUT_DIR)
        plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=200, bbox_inches="tight")


# =========================
# 1) VERİ OKU (OTOMATİK YIL KOLONLARI)
# =========================
def load_worldbank_timeseries_auto_years(
    file_path: str,
    country_code: str,
    year_min: int = 1960,
    year_max: int = 2024
) -> pd.DataFrame:
    def _clean_cols(df: pd.DataFrame) -> pd.DataFrame:
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.strip()
            .str.strip('"')
        )
        return df

    try:
        df = pd.read_csv(file_path, encoding="utf-8-sig")
        df = _clean_cols(df)
    except Exception:
        df = None

    if df is None or "Country Code" not in df.columns:
        df = pd.read_csv(file_path, skiprows=4, encoding="utf-8-sig")
        df = _clean_cols(df)

    if "Country Code" not in df.columns:
        raise ValueError("'Country Code' kolonu bulunamadı. CSV header/format kontrol et.")

    df["Country Code"] = df["Country Code"].astype(str).str.strip().str.strip('"')
    row = df[df["Country Code"] == country_code].copy()
    if row.empty:
        sample_codes = df["Country Code"].dropna().unique()[:10].tolist()
        raise ValueError(f"{country_code} bulunamadı. Örnek kodlar: {sample_codes}")

    year_cols = []
    for c in df.columns:
        cs = str(c).strip().strip('"')
        if cs.isdigit():
            y = int(cs)
            if year_min <= y <= year_max:
                year_cols.append(cs)

    if not year_cols:
        raise ValueError("Hiç yıl kolonu bulunamadı (1960-2024 aralığında).")

    year_cols = sorted(year_cols, key=lambda x: int(x))
    series = pd.to_numeric(row[year_cols].iloc[0], errors="coerce")

    ts = (
        pd.DataFrame({"Year": series.index.astype(int), "Value": series.values})
        .sort_values("Year")
        .reset_index(drop=True)
    )
    return ts


# =========================
# 2) VERİ SEÇİMİ
# =========================
def select_timespan(ts: pd.DataFrame, mode="contiguous", last_n_years=25) -> pd.DataFrame:
    df = ts.copy().sort_values("Year").reset_index(drop=True)

    if mode == "last_n":
        max_year = int(df["Year"].max())
        min_year = max_year - (last_n_years - 1)
        out = df[(df["Year"] >= min_year) & (df["Year"] <= max_year)].copy()
        return out.reset_index(drop=True)

    if mode == "contiguous":
        good = df.dropna(subset=["Value"]).copy()
        if good.empty:
            raise ValueError("Tüm değerler NaN. Ülke/indikatör kontrol et.")

        years = good["Year"].values.astype(int)

        segments = []
        start_idx = 0
        for i in range(1, len(years)):
            if years[i] != years[i - 1] + 1:
                segments.append((start_idx, i - 1))
                start_idx = i
        segments.append((start_idx, len(years) - 1))

        best = max(segments, key=lambda t: (t[1] - t[0] + 1))
        seg_years = years[best[0]: best[1] + 1]

        out = df[df["Year"].isin(seg_years)].copy()
        return out.sort_values("Year").reset_index(drop=True)

    raise ValueError("mode: contiguous | last_n")


# =========================
# 3) TEMİZLEME
# =========================
def fill_missing(ts: pd.DataFrame, method="interpolate") -> pd.DataFrame:
    out = ts.copy()
    if method == "interpolate":
        out["Value"] = out["Value"].interpolate(limit_direction="both")
    elif method == "ffill":
        out["Value"] = out["Value"].ffill()
    elif method == "bfill":
        out["Value"] = out["Value"].bfill()
    else:
        raise ValueError("method: interpolate | ffill | bfill")
    return out

def handle_outliers_iqr(ts: pd.DataFrame, k: float = 1.5, strategy="clip") -> pd.DataFrame:
    out = ts.copy()
    q1 = out["Value"].quantile(0.25)
    q3 = out["Value"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    if strategy == "clip":
        out["Value"] = out["Value"].clip(lower, upper)
    elif strategy == "remove":
        out = out[(out["Value"] >= lower) & (out["Value"] <= upper)].reset_index(drop=True)
    else:
        raise ValueError("strategy: clip | remove")

    return out


# =========================
# 4) METRİKLER
# =========================
def metrics(y_true, y_pred) -> dict:
    r2 = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    denom = np.where(np.abs(y_true) < 1e-9, 1e-9, np.abs(y_true))
    mape = float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)
    return {"R2": float(r2), "RMSE": rmse, "MAE": mae, "MAPE(%)": mape}

def print_eval_table(name, years, y_true, y_pred):
    print(f"\n===== {name} (TEST) =====")
    for yr, yt, yp in zip(years, y_true, y_pred):
        print(f"{yr} | Gerçek: {yt:.3f} | Tahmin: {yp:.3f}")
    m = metrics(y_true, y_pred)
    print("Metrikler:", m)
    return m


# =========================
# 5) NAIVE BASELINE (OPSİYONEL)
# =========================
def model_naive(ts: pd.DataFrame, test_size: int):
    y = ts["Value"].values
    y_train, y_test = y[:-test_size], y[-test_size:]
    last = float(y_train[-1])
    y_pred = np.array([last] * len(y_test))
    test_years = ts["Year"].iloc[-test_size:].tolist()
    m = print_eval_table("Naive (Last Value)", test_years, y_test, y_pred)
    return m


# =========================
# 6) ARIMA
# =========================
def safe_test_size(n: int, min_test_size: int = 3) -> int:
    if n < 12:
        return max(2, min_test_size - 1)
    return max(min_test_size, min(5, n // 4))

def _to_yearly_datetime_index(ts: pd.DataFrame) -> pd.Series:
    df = ts.copy()
    df["Year"] = pd.to_datetime(df["Year"].astype(int).astype(str) + "-01-01")
    s = df.set_index("Year")["Value"].asfreq("YS")  # Year Start
    return s

def fit_best_arima(train_s: pd.Series, p_max=3, d_max=2, q_max=3):
    best_order, best_aic, best_fit = None, np.inf, None

    for p in range(0, p_max + 1):
        for d in range(0, d_max + 1):
            for q in range(0, q_max + 1):
                if p == 0 and d == 0 and q == 0:
                    continue
                try:
                    fit = ARIMA(
                        train_s,
                        order=(p, d, q),
                        enforce_stationarity=False,
                        enforce_invertibility=False
                    ).fit(method_kwargs={"maxiter": 2000})

                    # yakınsamayanı ele
                    if not fit.mle_retvals.get("converged", True):
                        continue

                    if fit.aic < best_aic:
                        best_aic = fit.aic
                        best_order = (p, d, q)
                        best_fit = fit
                except Exception:
                    pass

    if best_fit is None:
        best_order = (1, 1, 0)
        best_fit = ARIMA(
            train_s, order=best_order,
            enforce_stationarity=False,
            enforce_invertibility=False
        ).fit(method_kwargs={"maxiter": 2000})
        best_aic = best_fit.aic

    return best_order, best_aic, best_fit

def model_arima(ts: pd.DataFrame, test_size: int):
    s = _to_yearly_datetime_index(ts)
    train_s = s.iloc[:-test_size]
    test_s  = s.iloc[-test_size:]

    order, aic, fit = fit_best_arima(train_s, p_max=P_MAX, d_max=D_MAX, q_max=Q_MAX)
    pred = fit.forecast(steps=test_size)

    test_years = [d.year for d in test_s.index.to_pydatetime()]
    m = print_eval_table(f"ARIMA{order} (AIC={aic:.2f})", test_years, test_s.values, pred.values)

    # fit nesnesini de döndürelim (diagnostics için)
    return order, m, fit, train_s, test_s, pred

def forecast_with_arima_full(ts: pd.DataFrame, order, future_years):
    s = _to_yearly_datetime_index(ts)
    fit = ARIMA(
        s,
        order=order,
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit()
    fc = fit.get_forecast(steps=len(future_years))
    mean = fc.predicted_mean
    ci = fc.conf_int(alpha=0.05)  # %95 CI
    # mean index datetime -> int year indexe çevir
    mean_vals = mean.values
    lower = ci.iloc[:, 0].values
    upper = ci.iloc[:, 1].values
    out = pd.DataFrame({"Year": future_years, "Forecast": mean_vals, "Lower95": lower, "Upper95": upper})
    return out


# =========================
# 7) POLİTİKA ÖNERİSİ
# =========================
def policy_suggestions_from_forecast(forecast_series: pd.Series, name: str):
    vals = forecast_series.values.astype(float)
    x = np.arange(len(vals))
    slope = np.polyfit(x, vals, 1)[0]

    start, end = float(vals[0]), float(vals[-1])
    change = end - start

    print(f"\n===== Politika Önerisi (Otomatik) | Model: {name} =====")
    print(f"{forecast_series.index[0]} tahmin: {start:.2f} | {forecast_series.index[-1]} tahmin: {end:.2f} | Değişim: {change:.2f}")

    if slope > 0.05:
        print("- Tahminler belirgin artış trendi gösteriyor.")
        print("- Öneri: Şebeke altyapı güçlendirme + depolama (batarya) teşvikleri.")
        print("- Öneri: Sanayi/konut enerji verimliliği programlarıyla talep azaltımı.")
    elif slope < -0.05:
        print("- Tahminler düşüş/gerileme eğilimi gösteriyor.")
        print("- Öneri: Teşvik mekanizmalarını güncelle + finansmanı kolaylaştır + lisans süreçlerini hızlandır.")
        print("- Öneri: Hedef odaklı alım garantileri / kapasite ihaleleri.")
    else:
        print("- Tahminler durağan/çok yavaş değişim gösteriyor.")
        print("- Öneri: Bölgesel teşvikler + kurulum maliyetini düşüren destekler.")
        print("- Öneri: Kamu binaları/belediyelerde yenilenebilir dönüşüm programları.")

    print("- Not: Bu öneriler tahmin trendine göre otomatik üretilmiştir.")


# =========================
# 8) RAPOR İÇİN EK GÖRSELLER
# =========================
def plot_missing_map(ts_full: pd.DataFrame):
    plt.figure(figsize=(11, 2.2))
    miss = ts_full["Value"].isna().astype(int).values
    plt.imshow(miss.reshape(1, -1), aspect="auto")
    plt.yticks([])
    plt.xticks(range(0, len(ts_full), max(1, len(ts_full)//10)), ts_full["Year"].iloc[::max(1, len(ts_full)//10)].tolist(), rotation=0)
    plt.title("Eksik Veri Haritası (1=NaN, 0=Var)")
    plt.tight_layout()
    savefig("01_missing_map.png")
    plt.show()

def plot_raw_vs_clean(ts_raw: pd.DataFrame, ts_clean: pd.DataFrame, title: str):
    plt.figure(figsize=(11, 4))
    plt.plot(ts_raw["Year"], ts_raw["Value"], marker="o", linestyle="--", label="Ham (seçilen dönem)")
    plt.plot(ts_clean["Year"], ts_clean["Value"], marker="o", label="Temiz (interp + IQR clip)")
    plt.title(title)
    plt.xlabel("Yıl")
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig("02_raw_vs_clean.png")
    plt.show()

def plot_iqr_outliers(ts_clean: pd.DataFrame, k=1.5):
    q1 = ts_clean["Value"].quantile(0.25)
    q3 = ts_clean["Value"].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    plt.figure(figsize=(11, 4))
    plt.plot(ts_clean["Year"], ts_clean["Value"], marker="o", label="Temiz Seri")
    plt.axhline(lower, linestyle="--", label="IQR Lower")
    plt.axhline(upper, linestyle="--", label="IQR Upper")
    plt.title("IQR Aykırı Değer Bandı")
    plt.xlabel("Yıl")
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig("03_iqr_outliers.png")
    plt.show()

def plot_train_test_split(ts: pd.DataFrame, test_size: int):
    plt.figure(figsize=(11, 4))
    plt.plot(ts["Year"].iloc[:-test_size], ts["Value"].iloc[:-test_size], marker="o", label="Train")
    plt.plot(ts["Year"].iloc[-test_size:], ts["Value"].iloc[-test_size:], marker="o", label="Test")
    plt.title("Train / Test Ayrımı")
    plt.xlabel("Yıl")
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig("04_train_test_split.png")
    plt.show()

def plot_test_forecast(ts_test_years: list, y_true: np.ndarray, y_pred: np.ndarray, order):
    plt.figure(figsize=(11, 4))
    plt.plot(ts_test_years, y_true, marker="o", label="Gerçek (Test)")
    plt.plot(ts_test_years, y_pred, marker="o", linestyle="--", label=f"ARIMA{order} Tahmin")
    plt.title("Test Dönemi: Gerçek vs Tahmin")
    plt.xlabel("Yıl")
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig("05_test_forecast.png")
    plt.show()

def plot_forecast_with_ci(ts: pd.DataFrame, fc_df: pd.DataFrame, order):
    plt.figure(figsize=(11, 5))
    plt.plot(ts["Year"], ts["Value"], marker="o", label="Gerçek")
    plt.plot(fc_df["Year"], fc_df["Forecast"], marker="o", linestyle="--", label=f"ARIMA{order} Forecast")
    plt.fill_between(fc_df["Year"], fc_df["Lower95"], fc_df["Upper95"], alpha=0.2, label="95% Güven Aralığı")
    plt.title(f"{COUNTRY_CODE} | ARIMA Tahmini + 95% Güven Aralığı")
    plt.xlabel("Yıl")
    plt.ylabel("Değer")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    savefig("06_forecast_with_ci.png")
    plt.show()

# ---- Residual diagnostics ----
def residual_diagnostics(fit, max_lag=10):
    """
    fit: statsmodels ARIMAResults (train fit)
    """
    resid = pd.Series(np.asarray(fit.resid)).dropna()

    # 1) Residual time plot
    plt.figure(figsize=(11, 4))
    plt.plot(resid.values, marker="o", linestyle="-")
    plt.axhline(0, linestyle="--")
    plt.title("Residuals (Artıklar) - Zaman Grafiği")
    plt.xlabel("Gözlem İndeksi")
    plt.ylabel("Residual")
    plt.grid(True)
    plt.tight_layout()
    savefig("07_residuals_time.png")
    plt.show()

    # 2) Histogram
    plt.figure(figsize=(11, 4))
    plt.hist(resid.values, bins=10)
    plt.title("Residuals Dağılımı (Histogram)")
    plt.xlabel("Residual")
    plt.ylabel("Frekans")
    plt.grid(True)
    plt.tight_layout()
    savefig("08_residual_hist.png")
    plt.show()

    # 3) ACF
    plt.figure(figsize=(11, 4))
    plot_acf(resid.values, lags=max_lag, ax=plt.gca())
    plt.title("Residuals ACF (Otokorelasyon)")
    plt.tight_layout()
    savefig("09_residual_acf.png")
    plt.show()

    # 4) QQ plot
    plt.figure(figsize=(6.5, 6.5))
    sm.qqplot(resid.values, line="45", ax=plt.gca())
    plt.title("Residuals QQ Plot (Normallik Kontrolü)")
    plt.tight_layout()
    savefig("10_residual_qq.png")
    plt.show()

    return resid

def ljung_box_report(resid: pd.Series, max_lag=10):
    """
    Ljung-Box: p-value yüksekse (genelde >0.05), artıklar 'beyaz gürültü' gibi -> iyi.
    """
    lb = acorr_ljungbox(resid.values, lags=list(range(1, max_lag + 1)), return_df=True)
    _ensure_outdir(OUTPUT_DIR)
    path = os.path.join(OUTPUT_DIR, "11_ljungbox.csv")
    lb.to_csv(path, index=True)
    print(f"[INFO] Ljung-Box raporu kaydedildi: {path}")
    print(lb.head(10))


# =========================
# 9) ANA AKIŞ
# =========================
def main():
    # 1) oku
    ts_full = load_worldbank_timeseries_auto_years(DATA_FILE, COUNTRY_CODE, YEAR_MIN, YEAR_MAX)

    print("Ham veri (ilk 10):")
    print(ts_full.head(10))
    print("Toplam yıl sayısı:", len(ts_full))
    print("Ham eksik sayısı:", int(ts_full["Value"].isna().sum()))

    # 2) missing map (rapor kanıtı)
    plot_missing_map(ts_full)

    # 3) seçili dönem (ham)
    ts_raw = select_timespan(ts_full, mode=SELECT_MODE, last_n_years=LAST_N_YEARS)

    # 4) temizle
    ts_clean = fill_missing(ts_raw, method="interpolate")
    ts_clean = handle_outliers_iqr(ts_clean, k=1.5, strategy="clip")

    print("\nSeçilen dönem:")
    print(f"{int(ts_clean['Year'].min())} - {int(ts_clean['Year'].max())} | nokta: {len(ts_clean)}")
    print("\nTemizlenmiş veri özet:")
    print(ts_clean.describe())

    # 5) rapor görselleri
    plot_raw_vs_clean(
        ts_raw,
        ts_clean,
        f"{COUNTRY_CODE} | Ham vs Temiz Seri ({int(ts_clean['Year'].min())}-{int(ts_clean['Year'].max())})"
    )
    plot_iqr_outliers(ts_clean, k=1.5)

    # 6) test size
    TEST_SIZE = safe_test_size(len(ts_clean), MIN_TEST_SIZE)
    print(f"\n[INFO] TEST_SIZE otomatik seçildi: {TEST_SIZE}")
    plot_train_test_split(ts_clean, TEST_SIZE)

    # 7) opsiyonel naive
    if USE_NAIVE_BASELINE:
        model_naive(ts_clean, TEST_SIZE)

    # 8) ARIMA train/test
    order, arima_m, train_fit, train_s, test_s, pred = model_arima(ts_clean, TEST_SIZE)

    test_years = [d.year for d in test_s.index.to_pydatetime()]
    plot_test_forecast(test_years, test_s.values, pred.values, order)

    # 9) gelecek tahmini + CI
    fc_df = forecast_with_arima_full(ts_clean, order, FORECAST_YEARS)
    plot_forecast_with_ci(ts_clean, fc_df, order)

    # 10) politika önerisi (ARIMA tek model)
    fc_series = pd.Series(fc_df["Forecast"].values, index=fc_df["Year"].values)
    policy_suggestions_from_forecast(fc_series, f"ARIMA{order}")

    # 11) Residual diagnostics (train fit üzerinden!)
    resid = residual_diagnostics(train_fit, max_lag=min(10, max(5, len(train_s)//2)))

    if RUN_LJUNG_BOX:
        ljung_box_report(resid, max_lag=min(10, max(5, len(resid)//2)))

    print(f"\n[INFO] Tüm görseller '{OUTPUT_DIR}/' klasörüne kaydedildi (SAVE_FIGS=True ise).")


if __name__ == "__main__":
    main()
