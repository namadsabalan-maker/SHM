# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import glob
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense

# تنظیمات اصلی صفحه
st.set_page_config(
    page_title="داشبورد پایش سلامت سازه",
    layout="wide",
    initial_sidebar_state="expanded"
)

# مسیرهای داده
DATA_FOLDER = 'data'
TEST_FOLDER = 'test_data'

# -----------------------------------------------------------
# توابع پردازش و تحلیل داده از فایل SHM3.py (ادغام شده)
# -----------------------------------------------------------

def process_raw_data(folder_path):
    all_raw_data = pd.DataFrame()
    day_folders = sorted(glob.glob(os.path.join(folder_path, '*/')))
    if not day_folders:
        st.warning(f"هیچ پوشه روزانه در پوشه '{folder_path}' یافت نشد.")
        return None
    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if os.path.exists(file_path):
            try:
                df = pd.read_csv(file_path)
                if {'x', 'y', 'z'}.issubset(df.columns):
                    df_accel = df[['x', 'y', 'z']].copy()
                    df_accel['date'] = date
                    all_raw_data = pd.concat([all_raw_data, df_accel], ignore_index=True)
            except Exception:
                continue
    return all_raw_data

def compute_magnitude(df_raw):
    df_magnitude = pd.DataFrame()
    if df_raw is None or df_raw.empty: return None
    for date, group in df_raw.groupby('date'):
        group['magnitude'] = np.sqrt(group['x']**2 + group['y']**2 + group['z']**2)
        df_magnitude = pd.concat([df_magnitude, group[['date', 'magnitude']]], ignore_index=True)
    return df_magnitude

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def apply_filter(df_magnitude, lowcut=0.1, highcut=10, fs=100):
    df_filtered = pd.DataFrame()
    if df_magnitude is None or df_magnitude.empty: return None
    for date, group in df_magnitude.groupby('date'):
        magnitude_filtered = butter_bandpass_filter(group['magnitude'].values, lowcut, highcut, fs)
        group['filtered_magnitude'] = magnitude_filtered
        df_filtered = pd.concat([df_filtered, group[['date', 'filtered_magnitude']]], ignore_index=True)
    return df_filtered

def extract_features(df_filtered):
    all_features = pd.DataFrame()
    if df_filtered is None or df_filtered.empty: return None
    for date, group in df_filtered.groupby('date'):
        signal = group['filtered_magnitude'].values
        features = {}
        features['mean'] = np.mean(signal)
        features['std'] = np.std(signal)
        features['variance'] = np.var(signal)
        features['max'] = np.max(signal)
        features['min'] = np.min(signal)
        features['median'] = np.median(signal)
        features['skewness'] = skew(signal)
        features['kurtosis'] = kurtosis(signal)
        N = len(signal)
        yf = np.fft.fft(signal)
        xf = np.fft.fftfreq(N, 1/100)
        psd = np.abs(yf[1:N//2])**2
        features['psd_mean'] = np.mean(psd)
        dominant_indices = np.argsort(psd)[-3:]
        dominant_freqs = xf[1:N//2][dominant_indices]
        for i, freq in enumerate(dominant_freqs):
            features[f'dom_freq_{i+1}'] = freq
        features['rms'] = np.sqrt(np.mean(signal**2))
        features['peak_to_peak'] = np.ptp(signal)
        hist, _ = np.histogram(signal, bins=100)
        hist_norm = hist / hist.sum()
        features['entropy'] = entropy(hist_norm, base=2)
        df_row = pd.DataFrame([features])
        df_row['date'] = date
        all_features = pd.concat([all_features, df_row], ignore_index=True)
    return all_features.set_index('date')

def mahalanobis_distance(x, mean, cov_inv):
    try:
        return np.sqrt(np.dot(np.dot((x - mean).values.reshape(1, -1), cov_inv), (x - mean).values.reshape(-1, 1)))[0][0]
    except (ValueError, np.linalg.LinAlgError):
        return np.inf

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(encoder)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

@st.cache_data
def run_analysis():
    st.info("در حال اجرای تحلیل داده‌ها. این ممکن است چند دقیقه طول بکشد...")
    df_raw = process_raw_data(DATA_FOLDER)
    if df_raw is None or df_raw.empty: return None, "هیچ داده‌ای برای پردازش یافت نشد."

    df_magnitude = compute_magnitude(df_raw)
    df_filtered = apply_filter(df_magnitude)
    df_features = extract_features(df_filtered)
    if df_features is None or df_features.empty: return None, "هیچ ویژگی‌ای استخراج نشد."

    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns, index=df_features.index)

    models = {}
    model_if = IsolationForest(random_state=42)
    model_if.fit(df_normalized)
    models['if'] = model_if
    
    mean_vec = df_normalized.mean()
    cov_matrix = df_normalized.cov()
    try:
        cov_inv = np.linalg.inv(cov_matrix)
        models['mahala'] = {'mean': mean_vec, 'cov_inv': cov_inv}
    except np.linalg.LinAlgError:
        pass
    
    input_dim = df_normalized.shape[1]
    model_ae = build_autoencoder(input_dim)
    model_ae.fit(df_normalized, df_normalized, epochs=50, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)
    models['ae'] = model_ae
    
    return {
        'df_raw': df_raw,
        'df_magnitude': df_magnitude,
        'df_filtered': df_filtered,
        'df_features': df_features,
        'df_normalized': df_normalized,
        'models': models,
        'scaler': scaler
    }, None

def run_test_and_get_results(models, scaler, df_features_train):
    day_folders = sorted(glob.glob(os.path.join(TEST_FOLDER, '*/')))
    if not day_folders:
        return pd.DataFrame(), pd.DataFrame(), "هیچ داده‌ای برای تست یافت نشد."
    
    test_results_df = pd.DataFrame(columns=['date', 'true_label', 'Isolation_Forest', 'Mahalanobis', 'Autoencoder', 'reason'])
    test_data = pd.DataFrame()

    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if not os.path.exists(file_path): continue
        try:
            true_label = 1 if ('frequency_shift' in date or 'spike' in date) else 0
            df_raw_test = pd.read_csv(file_path)
            if not {'x', 'y', 'z'}.issubset(df_raw_test.columns): continue
            
            df_raw_test['magnitude'] = np.sqrt(df_raw_test['x']**2 + df_raw_test['y']**2 + df_raw_test['z']**2)
            df_raw_test['filtered_magnitude'] = butter_bandpass_filter(df_raw_test['magnitude'].values, 0.1, 10, 100)
            df_raw_test['date'] = date
            test_data = pd.concat([test_data, df_raw_test], ignore_index=True)

            test_features = extract_features(df_raw_test[['filtered_magnitude', 'magnitude', 'date']].dropna())
            if test_features is None or test_features.empty: continue
            
            df_normalized_test = pd.DataFrame(scaler.transform(test_features), columns=test_features.columns)

            if_prediction = 1 if models['if'].predict(df_normalized_test)[0] == -1 else 0
            mahala_prediction = 0
            if 'mahala' in models:
                mahala_score = mahalanobis_distance(df_normalized_test.iloc[0], models['mahala']['mean'], models['mahala']['cov_inv'])
                mahala_prediction = 1 if mahala_score > 1.0 else 0

            ae_prediction = 0
            if 'ae' in models:
                ae_reconstruction_error = np.mean(np.power(df_normalized_test - models['ae'].predict(df_normalized_test, verbose=0), 2), axis=1)[0]
                ae_prediction = 1 if ae_reconstruction_error > 0.5 else 0

            anomaly_reason = ""
            if if_prediction == 1 or mahala_prediction == 1 or ae_prediction == 1:
                df_features_train_scaled = pd.DataFrame(scaler.transform(df_features_train), columns=df_features_train.columns)
                mean_normal = df_features_train_scaled.mean()
                std_normal = df_features_train_scaled.std()
                deviations = np.abs(df_normalized_test.iloc[0] - mean_normal) / std_normal
                top_deviated_features = deviations.nlargest(3)
                reason_parts = [f"{feature}: {test_features[feature].iloc[0]:.2f}" for feature, _ in top_deviated_features.items()]
                anomaly_reason = " | ".join(reason_parts)
            
            new_row = pd.DataFrame([{
                'date': date,
                'true_label': true_label,
                'Isolation_Forest': if_prediction,
                'Mahalanobis': mahala_prediction,
                'Autoencoder': ae_prediction,
                'reason': anomaly_reason
            }])
            test_results_df = pd.concat([test_results_df, new_row], ignore_index=True)
        except Exception:
            continue
    return test_results_df, test_data, None

# -----------------------------------------------------------
# رابط کاربری Streamlit
# -----------------------------------------------------------

# اجرای تحلیل داده و ذخیره در کش
analysis_results, analysis_error = run_analysis()

# عنوان اصلی
st.title("داشبورد پایش سلامت سازه پل کابلی تبریز")
st.markdown("---")

if analysis_error:
    st.error(f"خطا در پردازش: {analysis_error}")
    st.stop()
    
# تست و ارزیابی
test_results_df, test_data, test_error = run_test_and_get_results(analysis_results['models'], analysis_results['scaler'], analysis_results['df_features'])
if test_error:
    st.error(f"خطا در تست: {test_error}")
    st.stop()

# بخش نمایش وضعیت تحلیل
st.header("خلاصه وضعیت تحلیل")
st.markdown("**نتایج تشخیص ناهنجاری روی داده‌های جدید:**")
if not test_results_df.empty:
    st.dataframe(test_results_df)
else:
    st.info("هیچ فایل نتایج تستی برای نمایش وجود ندارد.")

# بخش نمایش نمودارها و تحلیل‌ها
st.markdown("---")
st.header("نمایش نمودارها و تحلیل‌های پروژه")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "داده‌های خام",
    "فیلتر و دامنه شتاب",
    "تحلیل ویژگی‌ها",
    "تحلیل حساسیت",
    "نتایج مدل‌ها",
    "گزارش تفسیری"
])

# تب ۱: داده‌های خام
with tab1:
    st.subheader("نمودارهای تاریخچه زمانی داده‌های خام")
    df_raw = analysis_results['df_raw']
    if df_raw is not None and not df_raw.empty:
        for date, group in df_raw.groupby('date'):
            st.markdown(f"**نمودار داده‌های خام برای روز {date}:**")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(group['x'], label='X-axis')
            ax.plot(group['y'], label='Y-axis')
            ax.plot(group['z'], label='Z-axis')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Acceleration')
            ax.set_title(f'Raw Accelerometer Data - {date}')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")
    else:
        st.info("هیچ داده خامی برای نمایش یافت نشد.")

# تب ۲: فیلتر و دامنه شتاب
with tab2:
    st.subheader("نمودارهای فیلتر و دامنه شتاب")
    df_magnitude = analysis_results['df_magnitude']
    df_filtered = analysis_results['df_filtered']
    
    if df_magnitude is not None and not df_magnitude.empty:
        st.markdown("**نمودارهای دامنه شتاب:**")
        for date, group in df_magnitude.groupby('date'):
            fig, ax = plt.subplots()
            ax.plot(group['magnitude'])
            ax.set_title(f'Accelerometer Magnitude - {date}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Magnitude')
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")
    
    if df_filtered is not None and not df_filtered.empty:
        st.markdown("**نمودارهای فیلترشده:**")
        for date, group in df_filtered.groupby('date'):
            fig, ax = plt.subplots()
            raw_mag = df_magnitude[df_magnitude['date'] == date]['magnitude'].values
            ax.plot(raw_mag, label='Raw Magnitude')
            ax.plot(group['filtered_magnitude'], label='Filtered Magnitude')
            ax.set_title(f'Raw vs Filtered Magnitude - {date}')
            ax.set_xlabel('Sample Index')
            ax.set_ylabel('Magnitude')
            ax.legend()
            st.pyplot(fig)
            plt.close(fig)
            st.markdown("---")
    if (df_magnitude is None or df_magnitude.empty) and (df_filtered is None or df_filtered.empty):
        st.info("هیچ نمودار دامنه یا فیلترشده‌ای یافت نشد.")

# تب ۳: تحلیل ویژگی‌ها
with tab3:
    st.subheader("تحلیل ویژگی‌ها")
    df_features = analysis_results['df_features']
    if df_features is not None and not df_features.empty:
        st.markdown("**ماتریس همبستگی ویژگی‌ها:**")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_features.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.info("نزدیکی مقادیر به ۱ یا -۱ نشان‌دهنده همبستگی قوی بین ویژگی‌هاست.")
    else:
        st.info("ماتریس همبستگی یافت نشد.")

# تب ۴: تحلیل حساسیت
with tab4:
    st.subheader("تحلیل حساسیت مدل‌ها")
    df_normalized = analysis_results['df_normalized']
    models = analysis_results['models']
    if df_normalized is not None and not df_normalized.empty and models:
        thresholds = np.arange(0.80, 1.0, 0.01)
        results = {'if': [], 'mahala': [], 'ae': []}
        
        if 'if' in models:
            if_scores = models['if'].decision_function(df_normalized)
            for threshold in thresholds:
                if_anomalies = np.sum(if_scores < np.percentile(if_scores, (1-threshold)*100))
                results['if'].append(if_anomalies)
        
        if 'mahala' in models:
            mahala_scores = df_normalized.apply(lambda row: mahalanobis_distance(row, models['mahala']['mean'], models['mahala']['cov_inv']), axis=1)
            for threshold in thresholds:
                mahala_anomalies = np.sum(mahala_scores > np.percentile(mahala_scores, threshold*100))
                results['mahala'].append(mahala_anomalies)
        
        if 'ae' in models:
            ae_reconstruction_error = np.mean(np.power(df_normalized - models['ae'].predict(df_normalized, verbose=0), 2), axis=1)
            for threshold in thresholds:
                ae_anomalies = np.sum(ae_reconstruction_error > np.percentile(ae_reconstruction_error, threshold*100))
                results['ae'].append(ae_anomalies)
                
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.set_title('Sensitivity Analysis - Number of Anomalies vs Threshold', fontsize=16)
        if 'if' in results:
            ax.plot(thresholds, results['if'], label='Isolation Forest')
        if 'mahala' in results:
            ax.plot(thresholds, results['mahala'], label='Mahalanobis Distance')
        if 'ae' in results:
            ax.plot(thresholds, results['ae'], label='Autoencoder')
        ax.set_xlabel('Threshold (%)')
        ax.set_ylabel('Number of Detected Anomalies')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        
        st.markdown("**گزارش متنی تحلیل حساسیت:**")
        st.info("این تحلیل نشان می‌دهد که با تغییر آستانه از 80% تا 99%، تعداد ناهنجاری‌های شناسایی شده توسط هر مدل چگونه تغییر می‌کند.")
        report_df = pd.DataFrame(results, index=[f'{int(t*100)}%' for t in thresholds])
        st.dataframe(report_df)
    else:
        st.info("تحلیل حساسیت قابل اجرا نیست. داده‌ها یا مدل‌ها موجود نیستند.")

# تب ۵: نتایج مدل‌ها
with tab5:
    st.subheader("نتایج تشخیص ناهنجاری")
    if not test_results_df.empty:
        for index, row in test_results_df.iterrows():
            date = row['date']
            test_data_group = test_data[test_data['date'] == date]
            if not test_data_group.empty:
                st.markdown(f"**نمودار تشخیص ناهنجاری برای {date}:**")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(test_data_group['filtered_magnitude'], label='Filtered Data')
                ax.text(0.05, 0.95, f'True Label: {"Anomaly" if row["true_label"] == 1 else "Normal"}', 
                         transform=ax.transAxes, fontsize=12, color='black', weight='bold')
                if row['Isolation_Forest'] == 1:
                    ax.scatter(np.arange(len(test_data_group))[::2000], test_data_group['filtered_magnitude'].values[::2000], color='red', s=100, label='IF Anomaly', marker='X')
                if row['Mahalanobis'] == 1:
                    ax.scatter(np.arange(len(test_data_group))[::2000], test_data_group['filtered_magnitude'].values[::2000], color='green', s=100, label='Mahalanobis Anomaly', marker='X')
                if row['Autoencoder'] == 1:
                    ax.scatter(np.arange(len(test_data_group))[::2000], test_data_group['filtered_magnitude'].values[::2000], color='purple', s=100, label='AE Anomaly', marker='X')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Magnitude')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("---")
    else:
        st.info("هیچ نمودار ناهنجاری‌ای برای نمایش وجود ندارد.")

# تب ۶: گزارش تفسیری
with tab6:
    st.subheader("گزارش تفسیری نهایی")
    if not test_results_df.empty:
        st.markdown("**خلاصه عملکرد مدل‌ها**")
        st.dataframe(test_results_df)

        st.markdown("---")
        st.markdown("**تحلیل عمیق ناهنجاری‌های شناسایی‌شده** 🕵️‍♂️")
        anomaly_dates = test_results_df[test_results_df['true_label'] == 1]
        if not anomaly_dates.empty:
            for index, row in anomaly_dates.iterrows():
                st.info(f"**تاریخ:** {row['date']}")
                st.markdown(f"**برچسب واقعی:** {'ناهنجاری' if row['true_label'] == 1 else 'عادی'}")
                st.markdown(f"**دلایل شناسایی توسط مدل‌ها:** {row['reason']}")
                st.markdown("---")
        else:
            st.info("هیچ ناهنجاری واقعی در داده‌های تست یافت نشد.")

    else:
        st.info("گزارش تفسیری یافت نشد. لطفاً ابتدا کد تحلیل را اجرا کنید.")

# -----------------------------------------------------------
# راهنمای استفاده
# -----------------------------------------------------------
st.sidebar.title("راهنمای استفاده")
st.sidebar.info(
    "1. این برنامه تمام مراحل تحلیل را به صورت خودکار اجرا می‌کند.\n"
    "2. تمام پوشه‌های `data` و `test_data` باید در همان مخزن گیت‌هاب قرار داشته باشند.\n"
    "3. برای اجرای این برنامه به صورت لوکال، از دستور زیر استفاده کنید:\n\n"
    "`streamlit run dashboard_final.py`"
)
st.sidebar.markdown("---")
st.sidebar.success("با این کار، داشبورد در مرورگر شما باز خواهد شد.")
