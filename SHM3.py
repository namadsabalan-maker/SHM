# -*- coding: utf-8 -*-

import os
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
import smtplib
from email.mime.text import MIMEText

# تنظیمات نمایش نمودارها
plt.style.use('seaborn-v0_8-whitegrid')

# مسیرهای ورودی و خروجی
DATA_FOLDER = 'data'
TEST_FOLDER = 'test_data'
OUTPUT_FOLDER = 'output'
RAW_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, 'raw_data')
MAGNITUDE_FOLDER = os.path.join(OUTPUT_FOLDER, 'acceleration_magnitude')
FILTERED_FOLDER = os.path.join(OUTPUT_FOLDER, 'filtered_data')
FEATURE_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, 'feature_data')
FEATURE_ANALYSIS_FOLDER = os.path.join(OUTPUT_FOLDER, 'feature_analysis')
NORMALIZED_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, 'normalized_data')
TRAINING_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'training_results')
SENSITIVITY_ANALYSIS_FOLDER = os.path.join(OUTPUT_FOLDER, 'sensitivity_analysis')
TEST_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'test_results')

# تنظیمات ایمیل (این مقادیر را با اطلاعات خود جایگزین کنید)
EMAIL_ADDRESS = 'j.sohafi@gmail.com'
EMAIL_PASSWORD = 'your_app_password' # از رمز عبور اصلی اکانت استفاده نکنید، رمز عبور برنامه بسازید
RECIPIENT_EMAIL = 'namadsabalan@gmail.com'

# -----------------------------------------------------------
# تابع ارسال ایمیل
# -----------------------------------------------------------
def send_alert_email(subject, body):
    """Sends an email notification."""
    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = EMAIL_ADDRESS
    msg['To'] = RECIPIENT_EMAIL
    
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as smtp:
            smtp.login(EMAIL_ADDRESS, EMAIL_PASSWORD)
            smtp.send_message(msg)
        print("ایمیل هشدار با موفقیت ارسال شد.")
    except Exception as e:
        print(f"خطا در ارسال ایمیل: {e}")

# ایجاد پوشه‌های خروجی در صورت عدم وجود
def create_directories():
    for folder in [OUTPUT_FOLDER, RAW_DATA_FOLDER, MAGNITUDE_FOLDER, FILTERED_FOLDER,
                   FEATURE_DATA_FOLDER, FEATURE_ANALYSIS_FOLDER, NORMALIZED_DATA_FOLDER,
                   TRAINING_RESULTS_FOLDER, SENSITIVITY_ANALYSIS_FOLDER, TEST_RESULTS_FOLDER]:
        os.makedirs(folder, exist_ok=True)

# -----------------------------------------------------------
# ۱. خواندن و رسم نمودار داده‌های خام
# -----------------------------------------------------------
def process_raw_data(folder_path):
    print("۱. در حال پردازش داده‌های خام...")
    all_raw_data = pd.DataFrame()
    day_folders = glob.glob(os.path.join(folder_path, '*/'))
    if not day_folders:
        print(f"هیچ پوشه روزانه در پوشه '{folder_path}' یافت نشد.")
        return None
    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if not os.path.exists(file_path):
            print(f"فایل Accelerometer.csv در پوشه {date} یافت نشد. به پوشه بعدی می‌رویم.")
            continue
        try:
            df = pd.read_csv(file_path)
            if {'x', 'y', 'z'}.issubset(df.columns):
                df_accel = df[['x', 'y', 'z']].copy()
                df_accel['date'] = date
                all_raw_data = pd.concat([all_raw_data, df_accel], ignore_index=True)
                plt.figure(figsize=(12, 6))
                plt.title(f'Raw Accelerometer Data - {date}', fontsize=16)
                plt.plot(df_accel['x'], label='X-axis')
                plt.plot(df_accel['y'], label='Y-axis')
                plt.plot(df_accel['z'], label='Z-axis')
                plt.xlabel('Sample Index')
                plt.ylabel('Acceleration')
                plt.legend()
                plt.savefig(os.path.join(RAW_DATA_FOLDER, f'raw_data_{date}.png'))
                plt.close()
            else:
                print(f"ستون‌های 'x', 'y', 'z' در فایل {file_path} یافت نشدند.")
        except Exception as e:
            print(f"خطا در خواندن یا پردازش فایل {file_path}: {e}")
            continue
    print("نمودارهای داده‌های خام با موفقیت ذخیره شدند.")
    return all_raw_data

# -----------------------------------------------------------
# ۲. محاسبه و رسم نمودار اندازه بردار شتاب
# -----------------------------------------------------------
def compute_magnitude(df_raw):
    print("۲. در حال محاسبه اندازه بردار شتاب...")
    df_magnitude = pd.DataFrame()
    for date, group in df_raw.groupby('date'):
        group['magnitude'] = np.sqrt(group['x']**2 + group['y']**2 + group['z']**2)
        df_magnitude = pd.concat([df_magnitude, group[['date', 'magnitude']]], ignore_index=True)
        plt.figure(figsize=(12, 6))
        plt.title(f'Accelerometer Magnitude - {date}', fontsize=16)
        plt.plot(group['magnitude'])
        plt.xlabel('Sample Index')
        plt.ylabel('Magnitude')
        plt.savefig(os.path.join(MAGNITUDE_FOLDER, f'magnitude_{date}.png'))
        plt.close()
    print("نمودارهای اندازه بردار شتاب با موفقیت ذخیره شدند.")
    return df_magnitude

# -----------------------------------------------------------
# ۳. اعمال فیلتر باند-پاس و رسم نمودار
# -----------------------------------------------------------
def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def apply_filter(df_magnitude, lowcut=0.1, highcut=10, fs=100):
    print("۳. در حال اعمال فیلتر باند-پاس...")
    df_filtered = pd.DataFrame()
    for date, group in df_magnitude.groupby('date'):
        magnitude_filtered = butter_bandpass_filter(group['magnitude'].values, lowcut, highcut, fs)
        group['filtered_magnitude'] = magnitude_filtered
        df_filtered = pd.concat([df_filtered, group[['date', 'filtered_magnitude']]], ignore_index=True)
        plt.figure(figsize=(12, 6))
        plt.title(f'Raw vs Filtered Magnitude - {date}', fontsize=16)
        plt.plot(group['magnitude'], label='Raw Magnitude')
        plt.plot(group['filtered_magnitude'], label='Filtered Magnitude')
        plt.xlabel('Sample Index')
        plt.ylabel('Magnitude')
        plt.legend()
        plt.savefig(os.path.join(FILTERED_FOLDER, f'filtered_data_{date}.png'))
        plt.close()
    print("نمودارهای داده‌های فیلترشده با موفقیت ذخیره شدند.")
    return df_filtered

# -----------------------------------------------------------
# ۴. استخراج ویژگی (Feature Extraction)
# -----------------------------------------------------------
def extract_features(df_filtered):
    print("۴. در حال استخراج ویژگی‌ها...")
    all_features = pd.DataFrame()
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
    all_features.to_csv(os.path.join(FEATURE_DATA_FOLDER, 'features.csv'), index=False)
    print("استخراج ویژگی‌ها با موفقیت انجام شد و در فایل features.csv ذخیره گردید.")
    return all_features.set_index('date')

# -----------------------------------------------------------
# ۵. تحلیل ویژگی‌ها
# -----------------------------------------------------------
def analyze_features(df_features):
    print("۵. در حال تحلیل ویژگی‌ها...")
    plt.figure(figsize=(12, 10))
    sns.heatmap(df_features.corr(), annot=True, fmt=".2f", cmap='coolwarm')
    plt.title('Correlation Matrix of Features', fontsize=16)
    plt.savefig(os.path.join(FEATURE_ANALYSIS_FOLDER, 'correlation_matrix.png'))
    plt.close()
    model_if = IsolationForest(random_state=42)
    model_if.fit(df_features)
    # Note: feature_importances_ is not a direct attribute of IsolationForest
    # The following block is a placeholder for feature importance analysis
    # on other models if needed. For IF, it's not a standard feature.
    # We will use a simple heuristic instead for deep analysis.
    # importances = model_if.estimators_[0].feature_importances_
    # df_importances = pd.DataFrame({'feature': df_features.columns, 'importance': importances})
    # df_importances = df_importances.sort_values('importance', ascending=False)
    # plt.figure(figsize=(12, 6))
    # sns.barplot(x='importance', y='feature', data=df_importances)
    # plt.title('Feature Importance', fontsize=16)
    # plt.savefig(os.path.join(FEATURE_ANALYSIS_FOLDER, 'feature_importance.png'))
    # plt.close()
    print("نمودارهای تحلیل ویژگی با موفقیت ذخیره شدند.")

# -----------------------------------------------------------
# ۶. نرمال‌سازی ویژگی‌ها
# -----------------------------------------------------------
def normalize_features(df_features):
    print("۶. در حال نرمال‌سازی ویژگی‌ها...")
    scaler = MinMaxScaler()
    df_normalized = pd.DataFrame(scaler.fit_transform(df_features), columns=df_features.columns, index=df_features.index)
    df_normalized.to_csv(os.path.join(NORMALIZED_DATA_FOLDER, 'normalized_features.csv'))
    print("ویژگی‌های نرمال‌شده در normalized_features.csv ذخیره شدند.")
    return df_normalized, scaler

# -----------------------------------------------------------
# ۷. آموزش مدل‌های تشخیص ناهنجاری
# -----------------------------------------------------------
def mahalanobis_distance(x, mean, cov_inv):
    return np.sqrt(np.dot(np.dot((x - mean).values.reshape(1, -1), cov_inv), (x - mean).values.reshape(-1, 1)))[0][0]

def build_autoencoder(input_dim):
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(64, activation='relu')(input_layer)
    encoder = Dense(32, activation='relu')(encoder)
    decoder = Dense(64, activation='relu')(encoder)
    output_layer = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

def train_models(df_normalized):
    print("۷. در حال آموزش مدل‌های تشخیص ناهنجاری...")
    models = {}
    model_if = IsolationForest(random_state=42)
    model_if.fit(df_normalized)
    models['if'] = model_if
    print("مدل Isolation Forest آموزش دید.")
    mean_vec = df_normalized.mean()
    cov_matrix = df_normalized.cov()
    try:
        cov_inv = np.linalg.inv(cov_matrix)
        models['mahala'] = {'mean': mean_vec, 'cov_inv': cov_inv}
        print("مدل Mahalanobis Distance آموزش دید.")
    except np.linalg.LinAlgError:
        print("خطا: ماتریس کوواریانس منفرد است. مدل Mahalanobis Distance آموزش داده نشد.")
    input_dim = df_normalized.shape[1]
    model_ae = build_autoencoder(input_dim)
    history = model_ae.fit(df_normalized, df_normalized, epochs=50, batch_size=32, shuffle=True, validation_split=0.1, verbose=0)
    models['ae'] = model_ae
    plt.figure(figsize=(10, 6))
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Autoencoder Training Loss', fontsize=16)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(os.path.join(TRAINING_RESULTS_FOLDER, 'autoencoder_loss.png'))
    plt.close()
    print("مدل Autoencoder آموزش دید.")
    with open(os.path.join(TRAINING_RESULTS_FOLDER, 'training_report.txt'), 'w', encoding='utf-8') as f:
        f.write("گزارش آموزش مدل‌های تشخیص ناهنجاری:\n")
        f.write("-------------------------------------\n\n")
        f.write("1. Isolation Forest:\n")
        f.write("این مدل با استفاده از ویژگی‌های نرمال‌شده آموزش دید.\n\n")
        f.write("2. Mahalanobis Distance:\n")
        f.write("این مدل بر اساس میانگین و ماتریس کوواریانس داده‌های نرمال‌شده آموزش دید.\n\n")
        f.write("3. Autoencoder:\n")
        f.write("مدل Autoencoder برای بازسازی ویژگی‌های نرمال‌شده آموزش داده شد. نمودار خطای آموزش نیز ذخیره شده است.\n")
    print("گزارش آموزش مدل‌ها با موفقیت ذخیره شد.")
    return models

# -----------------------------------------------------------
# ۸. تحلیل حساسیت (Sensitivity Analysis)
# -----------------------------------------------------------
def sensitivity_analysis(models, df_normalized):
    print("۸. در حال انجام تحلیل حساسیت...")
    thresholds = np.arange(0.80, 1.0, 0.01)
    results = {
        'if': [],
        'mahala': [],
        'ae': []
    }
    if 'if' in models:
        if_scores = models['if'].decision_function(df_normalized)
    else:
        if_scores = []
        
    if 'mahala' in models:
        mahala_scores = df_normalized.apply(lambda row: mahalanobis_distance(row, models['mahala']['mean'], models['mahala']['cov_inv']), axis=1)
    else:
        mahala_scores = []
        
    if 'ae' in models:
        ae_reconstruction_error = np.mean(np.power(df_normalized - models['ae'].predict(df_normalized, verbose=0), 2), axis=1)
    else:
        ae_reconstruction_error = []
        
    for threshold in thresholds:
        if 'if' in models:
            if_anomalies = np.sum(if_scores < np.percentile(if_scores, (1-threshold)*100))
            results['if'].append(if_anomalies)
        else:
            results['if'].append(0)
            
        if 'mahala' in models:
            mahala_anomalies = np.sum(mahala_scores > np.percentile(mahala_scores, threshold*100))
            results['mahala'].append(mahala_anomalies)
        else:
            results['mahala'].append(0)
            
        if 'ae' in models:
            ae_anomalies = np.sum(ae_reconstruction_error > np.percentile(ae_reconstruction_error, threshold*100))
            results['ae'].append(ae_anomalies)
        else:
            results['ae'].append(0)
            
    plt.figure(figsize=(12, 6))
    plt.title('Sensitivity Analysis - Number of Anomalies vs Threshold', fontsize=16)
    plt.plot(thresholds, results['if'], label='Isolation Forest')
    if 'mahala' in models:
        plt.plot(thresholds, results['mahala'], label='Mahalanobis Distance')
    plt.plot(thresholds, results['ae'], label='Autoencoder')
    plt.xlabel('Threshold (%)')
    plt.ylabel('Number of Detected Anomalies')
    plt.legend()
    plt.savefig(os.path.join(SENSITIVITY_ANALYSIS_FOLDER, 'sensitivity_analysis.png'))
    plt.close()
    with open(os.path.join(SENSITIVITY_ANALYSIS_FOLDER, 'sensitivity_report.txt'), 'w', encoding='utf-8') as f:
        f.write("گزارش تحلیل حساسیت مدل‌ها:\n")
        f.write("-------------------------------------\n\n")
        f.write("این تحلیل نشان می‌دهد که با تغییر آستانه از 80% تا 99%، تعداد ناهنجاری‌های شناسایی شده توسط هر مدل چگونه تغییر می‌کند.\n")
        f.write(pd.DataFrame(results, index=[f'{int(t*100)}%' for t in thresholds]).to_string())
    print("تحلیل حساسیت با موفقیت انجام شد.")

# -----------------------------------------------------------
# ۹ و ۱۰. تست و ارزیابی مدل‌ها روی داده‌های جدید با برچسب
# -----------------------------------------------------------
def test_and_evaluate_models(models, scaler, df_features):
    print("۹. در حال تست و ارزیابی مدل‌ها روی داده‌های جدید...")
    day_folders = glob.glob(os.path.join(TEST_FOLDER, '*/'))
    if not day_folders:
        print(f"هیچ پوشه روزانه در پوشه '{TEST_FOLDER}' یافت نشد.")
        return None
    
    test_results_df = pd.DataFrame(columns=['date', 'true_label', 'Isolation_Forest', 'Mahalanobis', 'Autoencoder', 'reason'])
    true_labels = []
    if_predictions = []
    mahala_predictions = []
    ae_predictions = []

    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if not os.path.exists(file_path):
            print(f"فایل Accelerometer.csv در پوشه تست {date} یافت نشد.")
            continue
        try:
            true_label = 1 if ('frequency_shift' in date or 'spike' in date) else 0
            true_labels.append(true_label)
            
            df_raw_test = pd.read_csv(file_path)
            if not {'x', 'y', 'z'}.issubset(df_raw_test.columns):
                print(f"ستون‌های 'x', 'y', 'z' در فایل {file_path} یافت نشدند.")
                continue
            
            df_raw_test['magnitude'] = np.sqrt(df_raw_test['x']**2 + df_raw_test['y']**2 + df_raw_test['z']**2)
            df_raw_test['filtered_magnitude'] = butter_bandpass_filter(df_raw_test['magnitude'].values, 0.1, 10, 100)
            
            test_features = {}
            signal = df_raw_test['filtered_magnitude'].values
            test_features['mean'] = np.mean(signal)
            test_features['std'] = np.std(signal)
            test_features['variance'] = np.var(signal)
            test_features['max'] = np.max(signal)
            test_features['min'] = np.min(signal)
            test_features['median'] = np.median(signal)
            test_features['skewness'] = skew(signal)
            test_features['kurtosis'] = kurtosis(signal)
            N = len(signal)
            yf = np.fft.fft(signal)
            xf = np.fft.fftfreq(N, 1/100)
            psd = np.abs(yf[1:N//2])**2
            test_features['psd_mean'] = np.mean(psd)
            dominant_indices = np.argsort(psd)[-3:]
            dominant_freqs = xf[1:N//2][dominant_indices]
            for i, freq in enumerate(dominant_freqs):
                test_features[f'dom_freq_{i+1}'] = freq
            test_features['rms'] = np.sqrt(np.mean(signal**2))
            test_features['peak_to_peak'] = np.ptp(signal)
            hist, _ = np.histogram(signal, bins=100)
            hist_norm = hist / hist.sum()
            test_features['entropy'] = entropy(hist_norm, base=2)
            
            df_features_test = pd.DataFrame([test_features])
            df_normalized_test = pd.DataFrame(scaler.transform(df_features_test), columns=df_features_test.columns)

            if_prediction = 1 if models['if'].predict(df_normalized_test)[0] == -1 else 0
            if_predictions.append(if_prediction)

            mahala_prediction = 0
            if 'mahala' in models:
                mahala_score = mahalanobis_distance(df_normalized_test.iloc[0], models['mahala']['mean'], models['mahala']['cov_inv'])
                mahala_prediction = 1 if mahala_score > 1.0 else 0
            mahala_predictions.append(mahala_prediction)

            ae_prediction = 0
            if 'ae' in models:
                ae_reconstruction_error = np.mean(np.power(df_normalized_test - models['ae'].predict(df_normalized_test, verbose=0), 2), axis=1)[0]
                ae_prediction = 1 if ae_reconstruction_error > 0.5 else 0
            ae_predictions.append(ae_prediction)

            # تحلیل علت ناهنجاری
            anomaly_reason = ""
            if if_prediction == 1 or mahala_prediction == 1 or ae_prediction == 1:
                df_features_train_scaled = pd.DataFrame(scaler.transform(df_features), columns=df_features.columns)
                
                # محاسبه انحراف از میانگین داده‌های نرمال
                mean_normal = df_features_train_scaled.mean()
                std_normal = df_features_train_scaled.std()
                
                deviations = np.abs(df_normalized_test.iloc[0] - mean_normal) / std_normal
                top_deviated_features = deviations.nlargest(3)
                
                reason_parts = []
                for feature, dev in top_deviated_features.items():
                    reason_parts.append(f"{feature}: {df_features_test[feature].iloc[0]:.2f}")
                
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

            plt.figure(figsize=(12, 6))
            plt.title(f'Detected Anomalies on Features - {date}', fontsize=16)
            plt.plot(df_raw_test['filtered_magnitude'], label='Filtered Data')
            plt.text(0.05, 0.95, f'True Label: {"Anomaly" if true_label == 1 else "Normal"}', 
                     transform=plt.gca().transAxes, fontsize=12, color='black', weight='bold')
            if if_prediction == 1:
                plt.scatter([len(df_raw_test) / 2], [df_raw_test['filtered_magnitude'].mean()], color='red', s=200, label='IF Predicted Anomaly', marker='X')
            if mahala_prediction == 1:
                plt.scatter([len(df_raw_test) / 2 + 50], [df_raw_test['filtered_magnitude'].mean()], color='green', s=200, label='Mahalanobis Predicted Anomaly', marker='X')
            if ae_prediction == 1:
                plt.scatter([len(df_raw_test) / 2 - 50], [df_raw_test['filtered_magnitude'].mean()], color='purple', s=200, label='AE Predicted Anomaly', marker='X')
            plt.xlabel('Sample Index')
            plt.ylabel('Magnitude')
            plt.legend()
            plt.savefig(os.path.join(TEST_RESULTS_FOLDER, f'anomalies_{date}.png'))
            plt.close()
            
            # بخش جدید: بررسی اجماع و ارسال ایمیل
            if if_prediction == 1 and mahala_prediction == 1 and ae_prediction == 1:
                subject = f"هشدار حیاتی ناهنجاری در پل - تاریخ: {date}"
                body = (f"یک ناهنجاری حیاتی در داده‌های پل شناسایی شده است.\n"
                        f"هر سه مدل (Isolation Forest، Mahalanobis، Autoencoder) ناهنجاری را تأیید کرده‌اند.\n"
                        f"دلایل احتمالی: {anomaly_reason}\n\n"
                        f"برای اطلاعات بیشتر به داشبورد مدیریتی مراجعه کنید.")
                send_alert_email(subject, body)


        except Exception as e:
            print(f"خطا در پردازش فایل تست {file_path}: {e}")
            continue

    test_results_df.to_csv(os.path.join(TEST_RESULTS_FOLDER, 'test_results.csv'), index=False)
    print("گزارش تست مدل‌ها و تعداد ناهنجاری‌ها ذخیره شد.")

    with open(os.path.join(TEST_RESULTS_FOLDER, 'evaluation_metrics.txt'), 'w', encoding='utf-8') as f:
        f.write("گزارش ارزیابی مدل‌ها روی داده‌های تست:\n")
        f.write("----------------------------------------\n\n")
        
        f.write("Isolation Forest:\n")
        cm_if = confusion_matrix(true_labels, if_predictions)
        f.write(f"Confusion Matrix:\n{cm_if}\n")
        f.write(f"Precision: {precision_score(true_labels, if_predictions):.2f}\n")
        f.write(f"Recall: {recall_score(true_labels, if_predictions):.2f}\n")
        f.write(f"F1-Score: {f1_score(true_labels, if_predictions):.2f}\n\n")
        
        f.write("Mahalanobis Distance:\n")
        cm_mahala = confusion_matrix(true_labels, mahala_predictions)
        f.write(f"Confusion Matrix:\n{cm_mahala}\n")
        f.write(f"Precision: {precision_score(true_labels, mahala_predictions):.2f}\n")
        f.write(f"Recall: {recall_score(true_labels, mahala_predictions):.2f}\n")
        f.write(f"F1-Score: {f1_score(true_labels, mahala_predictions):.2f}\n\n")
        
        f.write("Autoencoder:\n")
        cm_ae = confusion_matrix(true_labels, ae_predictions)
        f.write(f"Confusion Matrix:\n{cm_ae}\n")
        f.write(f"Precision: {precision_score(true_labels, ae_predictions):.2f}\n")
        f.write(f"Recall: {recall_score(true_labels, ae_predictions):.2f}\n")
        f.write(f"F1-Score: {f1_score(true_labels, ae_predictions):.2f}\n\n")
    print("گزارش معیارهای ارزیابی با موفقیت ذخیره شد.")

    return test_results_df

# -----------------------------------------------------------
# ۱۱. تولید گزارش تفسیری نهایی
# -----------------------------------------------------------
def generate_interpretation_report(test_results_df, eval_metrics_file):
    print("در حال تولید گزارش تفسیری...")
    with open(eval_metrics_file, 'r', encoding='utf-8') as f:
        metrics_report = f.read()
    with open(os.path.join(TEST_RESULTS_FOLDER, 'interpretation_report.txt'), 'w', encoding='utf-8') as f:
        f.write("گزارش تفسیری نتایج تشخیص ناهنجاری\n")
        f.write("=====================================\n\n")
        
        f.write("۱. خلاصه عملکرد مدل‌ها\n")
        f.write("----------------------\n")
        f.write(metrics_report)
        f.write("\n")
        
        f.write("۲. تحلیل عمیق ناهنجاری‌های شناسایی‌شده 🕵️‍♂️\n")
        f.write("-----------------------------------------\n")
        
        anomaly_dates = test_results_df[test_results_df['true_label'] == 1]
        if not anomaly_dates.empty:
            f.write("داده‌های دارای ناهنجاری واقعی و دلایل احتمالی:\n")
            for index, row in anomaly_dates.iterrows():
                f.write(f"- تاریخ: {row['date']}\n")
                f.write(f"  - برچسب واقعی: {'ناهنجاری' if row['true_label'] == 1 else 'عادی'}\n")
                f.write(f"  - دلایل شناسایی توسط مدل‌ها: {row['reason']}\n\n")
            f.write("\n")
        
        f.write("۳. تحلیل موارد خطا (False Alarms & Missed Detections)\n")
        f.write("---------------------------------------------------\n")
        for model_name in ['Isolation_Forest', 'Mahalanobis', 'Autoencoder']:
            f.write(f"\n**{model_name}**\n")
            
            fp_df = test_results_df[(test_results_df['true_label'] == 0) & (test_results_df[model_name] == 1)]
            if not fp_df.empty:
                f.write("   - **هشدارهای کاذب (False Alarms):**\n")
                for index, row in fp_df.iterrows():
                    f.write(f"     - تاریخ: {row['date']} | دلایل شناسایی: {row['reason']}\n")
            else:
                f.write("   - هیچ هشدار کاذبی شناسایی نشد.\n")
            
            fn_df = test_results_df[(test_results_df['true_label'] == 1) & (test_results_df[model_name] == 0)]
            if not fn_df.empty:
                f.write("   - **ناهنجاری‌های شناسایی‌نشده (Missed Detections):**\n")
                for index, row in fn_df.iterrows():
                    f.write(f"     - تاریخ: {row['date']} | دلایل احتمالی: {row['reason']}\n")
            else:
                f.write("   - تمام ناهنجاری‌های واقعی شناسایی شدند.\n")
    print("گزارش تفسیری با موفقیت تولید و ذخیره شد.")

# -----------------------------------------------------------
# تابع اصلی
# -----------------------------------------------------------
def main():
    create_directories()
    df_raw = process_raw_data(DATA_FOLDER)
    if df_raw is None or df_raw.empty:
        print("هیچ داده‌ای برای پردازش یافت نشد. برنامه متوقف می‌شود.")
        return
    df_magnitude = compute_magnitude(df_raw)
    df_filtered = apply_filter(df_magnitude)
    df_features = extract_features(df_filtered)
    if df_features.empty:
        print("هیچ ویژگی‌ای استخراج نشد. برنامه متوقف می‌شود.")
        return
    analyze_features(df_features)
    df_normalized, scaler = normalize_features(df_features)
    models = train_models(df_normalized)
    sensitivity_analysis(models, df_normalized)
    
    # Passing df_features to test_and_evaluate_models for feature analysis
    test_results_df = test_and_evaluate_models(models, scaler, df_features)
    
    if test_results_df is not None:
        generate_interpretation_report(test_results_df, os.path.join(TEST_RESULTS_FOLDER, 'evaluation_metrics.txt'))
        
    print("\nبرنامه با موفقیت به پایان رسید. نتایج در پوشه 'output' ذخیره شده‌اند.")

if __name__ == '__main__':
    main()
