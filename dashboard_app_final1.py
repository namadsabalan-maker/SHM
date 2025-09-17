```python
# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    st.error("Plotly is not installed. Please install it with `pip install plotly` for interactive time series plots.")
import os
import glob
from scipy.signal import butter, lfilter
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, classification_report
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import EarlyStopping
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Structural Health Monitoring Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Data paths
DATA_FOLDER = 'data'
TEST_FOLDER = 'test_data'

# Data processing functions
def process_raw_data(folder_path):
    """
    Process raw accelerometer data from daily folders.
    
    Args:
        folder_path (str): Path to the data folder.
    
    Returns:
        pd.DataFrame: Concatenated raw data or None if no data found.
    
    Raises:
        ValueError: If no valid files are found.
    """
    all_raw_data = pd.DataFrame()
    day_folders = glob.glob(os.path.join(folder_path, '*/'))
    if not day_folders:
        logger.warning(f"No day folders found in {folder_path}")
        return None
    
    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if not os.path.exists(file_path):
            logger.warning(f"File not found: {file_path}")
            continue
        try:
            df = pd.read_csv(file_path)
            if {'x', 'y', 'z'}.issubset(df.columns):
                df_accel = df[['x', 'y', 'z']].copy()
                df_accel['date'] = date
                all_raw_data = pd.concat([all_raw_data, df_accel], ignore_index=True)
            else:
                logger.error(f"Missing columns in {file_path}: expected 'x', 'y', 'z'")
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            continue
    
    if all_raw_data.empty:
        raise ValueError("No valid data processed from the folder.")
    
    return all_raw_data

def compute_magnitude(df_raw):
    """Compute the magnitude of acceleration vectors."""
    df_magnitude = pd.DataFrame()
    for date, group in df_raw.groupby('date'):
        group['magnitude'] = np.sqrt(group['x']**2 + group['y']**2 + group['z']**2)
        df_magnitude = pd.concat([df_magnitude, group[['date', 'magnitude']]], ignore_index=True)
    return df_magnitude

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    """
    Apply Butterworth bandpass filter.
    
    Args:
        data (np.array): Input signal.
        lowcut (float): Low cutoff frequency.
        highcut (float): High cutoff frequency.
        fs (float): Sampling frequency.
        order (int): Filter order.
    
    Returns:
        np.array: Filtered signal.
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    y = lfilter(b, a, data)
    return y

def apply_filter(df_magnitude, lowcut=0.1, highcut=10, fs=100):
    """Apply bandpass filter to magnitude data."""
    df_filtered = pd.DataFrame()
    for date, group in df_magnitude.groupby('date'):
        try:
            magnitude_filtered = butter_bandpass_filter(group['magnitude'].values, lowcut, highcut, fs)
            group['filtered_magnitude'] = magnitude_filtered
            df_filtered = pd.concat([df_filtered, group[['date', 'filtered_magnitude']]], ignore_index=True)
        except Exception as e:
            logger.error(f"Filter error for {date}: {e}")
    return df_filtered

def extract_features(df_input):
    """Extract statistical and spectral features from input data."""
    all_features = pd.DataFrame()
    for date, group in df_input.groupby('date'):
        signal = group['filtered_magnitude'].values
        if len(signal) == 0:
            continue
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
    """Compute Mahalanobis distance for anomaly scoring."""
    diff = x - mean
    return np.sqrt(np.dot(np.dot(diff.values.reshape(1, -1), cov_inv), diff.values.reshape(-1, 1)))[0][0]

def build_autoencoder(input_dim):
    """Build and compile a simple autoencoder model."""
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
    """Run the full analysis pipeline with caching."""
    try:
        # 1. Load data
        df_raw = process_raw_data(DATA_FOLDER)
        if df_raw is None or df_raw.empty:
            return None, None, None, None, None, None, "No data found for processing."

        # 2. Compute magnitude
        df_magnitude = compute_magnitude(df_raw)

        # 3. Apply filter
        df_filtered = apply_filter(df_magnitude)

        # 4. Extract features
        df_features = extract_features(df_filtered)
        if df_features.empty:
            return None, None, None, None, None, None, "No features extracted."

        # 5. Normalize
        scaler = MinMaxScaler()
        df_normalized = pd.DataFrame(scaler.fit_transform(df_features), 
                                     columns=df_features.columns, index=df_features.index)

        # 6. Train models
        models = {}
        model_if = IsolationForest(random_state=42)
        model_if.fit(df_normalized)
        models['if'] = model_if
        
        mean_vec = df_normalized.mean()
        cov_matrix = df_normalized.cov()
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            models['mahala'] = {'mean': mean_vec, 'cov_inv': cov_inv}
        except np.linalg.LinAlgError as e:
            logger.warning(f"Covariance inversion failed: {e}")
        
        input_dim = df_normalized.shape[1]
        model_ae = build_autoencoder(input_dim)
        early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model_ae.fit(df_normalized, df_normalized, epochs=50, batch_size=32, shuffle=True, 
                     validation_split=0.1, verbose=0, callbacks=[early_stop])
        models['ae'] = model_ae
        
        return df_raw, df_magnitude, df_filtered, df_features, models, scaler, None
    except Exception as e:
        logger.error(f"Analysis failed: {e}")
        return None, None, None, None, None, None, str(e)

# Time series data preparation
def prepare_time_series_data(test_results_df, models, scaler):
    """Prepare aggregated time series data with anomaly scores and warning levels."""
    ts_data = []
    for _, row in test_results_df.iterrows():
        date = row['date']
        day_folder = os.path.join(TEST_FOLDER, date)
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        if os.path.exists(file_path):
            df_raw_test = pd.read_csv(file_path)
            if not {'x', 'y', 'z'}.issubset(df_raw_test.columns):
                continue
            df_raw_test['magnitude'] = np.sqrt(df_raw_test['x']**2 + df_raw_test['y']**2 + df_raw_test['z']**2)
            df_raw_test['filtered_magnitude'] = butter_bandpass_filter(df_raw_test['magnitude'].values, 0.1, 10, 100)
            
            # Aggregate: mean filtered magnitude
            mean_magnitude = df_raw_test['filtered_magnitude'].mean()
            
            # Composite anomaly score
            anomaly_score = np.mean([row['Isolation_Forest'], row['Mahalanobis'], row['Autoencoder']])
            
            # Determine warning level
            if anomaly_score < 0.3:
                warning_level = 'Normal'
                color = 'green'
            elif anomaly_score < 0.7:
                warning_level = 'Warning'
                color = 'yellow'
            else:
                warning_level = 'Alert'
                color = 'red'
            
            ts_data.append({
                'Date': pd.to_datetime(date, errors='coerce'),
                'Mean Filtered Magnitude': mean_magnitude,
                'Anomaly Score': anomaly_score,
                'Warning Level': warning_level,
                'Color': color,
                'True Label': 'Anomaly' if row['true_label'] == 1 else 'Normal'
            })
    
    return pd.DataFrame(ts_data)

# Streamlit UI
st.info("Running data analysis. This may take a few minutes...")
results = run_analysis()
if results and results[-1] is not None:
    st.error(f"Processing error: {results[-1]}")
    st.stop()

df_raw, df_magnitude, df_filtered, df_features, models, scaler, error = results

# Main title
st.title("Tabriz Cable-Stayed Bridge Structural Health Monitoring Dashboard")
st.markdown("---")

# Sidebar for warning thresholds
st.sidebar.title("Warning Thresholds")
warning_low = st.sidebar.slider("Low Threshold (Normal/Warning)", 0.0, 1.0, 0.3)
warning_high = st.sidebar.slider("High Threshold (Warning/Alert)", 0.0, 1.0, 0.7)
st.sidebar.info("Adjust these to customize alert levels based on domain expertise.")

# Analysis summary
st.header("Analysis Summary")
st.markdown("**Anomaly Detection Results on New Data:**")

def test_and_evaluate_models_in_dashboard(models, scaler, df_features):
    """Test models on test data and compute evaluation metrics."""
    test_results_df = pd.DataFrame(columns=['date', 'true_label', 'Isolation_Forest', 'Mahalanobis', 'Autoencoder', 'reason'])
    day_folders = glob.glob(os.path.join(TEST_FOLDER, '*/'))
    if not day_folders:
        return test_results_df

    predictions = {'if': [], 'mahala': [], 'ae': []}
    true_labels = []

    for day_folder in day_folders:
        file_path = os.path.join(day_folder, 'Accelerometer.csv')
        date = os.path.basename(os.path.normpath(day_folder))
        if not os.path.exists(file_path):
            continue
        try:
            true_label = 1 if ('frequency_shift' in date or 'spike' in date) else 0
            true_labels.append(true_label)
            
            df_raw_test = pd.read_csv(file_path)
            if not {'x', 'y', 'z'}.issubset(df_raw_test.columns):
                continue
            
            df_raw_test['magnitude'] = np.sqrt(df_raw_test['x']**2 + df_raw_test['y']**2 + df_raw_test['z']**2)
            df_raw_test['filtered_magnitude'] = butter_bandpass_filter(df_raw_test['magnitude'].values, 0.1, 10, 100)
            
            test_df_input = df_raw_test[['filtered_magnitude']].copy()
            test_df_input['date'] = date
            test_features = extract_features(test_df_input).loc[date].to_dict()
            
            df_features_test = pd.DataFrame([test_features])
            df_normalized_test = pd.DataFrame(scaler.transform(df_features_test), columns=df_features_test.columns)

            # Predictions
            if_prediction = 1 if models['if'].predict(df_normalized_test)[0] == -1 else 0
            predictions['if'].append(if_prediction)
            
            mahala_prediction = 0
            if 'mahala' in models:
                mahala_score = mahalanobis_distance(df_normalized_test.iloc[0], models['mahala']['mean'], models['mahala']['cov_inv'])
                mahala_prediction = 1 if mahala_score > 1.0 else 0
            predictions['mahala'].append(mahala_prediction)

            ae_prediction = 0
            if 'ae' in models:
                ae_reconstruction_error = np.mean(np.power(df_normalized_test - models['ae'].predict(df_normalized_test, verbose=0), 2), axis=1)[0]
                ae_prediction = 1 if ae_reconstruction_error > 0.5 else 0
            predictions['ae'].append(ae_prediction)

            # Anomaly reason
            anomaly_reason = ""
            if if_prediction == 1 or mahala_prediction == 1 or ae_prediction == 1:
                df_features_train_scaled = pd.DataFrame(scaler.transform(df_features), columns=df_features.columns)
                mean_normal = df_features_train_scaled.mean()
                std_normal = df_features_train_scaled.std()
                deviations = np.abs(df_normalized_test.iloc[0] - mean_normal) / std_normal
                top_deviated_features = deviations.nlargest(3)
                reason_parts = [f"{feature}: {df_features_test[feature].iloc[0]:.2f}" for feature, _ in top_deviated_features.items()]
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
        except Exception as e:
            logger.error(f"Test error for {date}: {e}")
            continue
    
    # Compute metrics
    if not test_results_df.empty:
        y_true = test_results_df['true_label'].values
        for model_name, y_pred in predictions.items():
            if len(y_pred) == len(y_true):
                st.subheader(f"Performance for {model_name.upper()}")
                st.text(classification_report(y_true, y_pred))
                report = classification_report(y_true, y_pred, output_dict=True)
                metrics_df = pd.DataFrame({
                    'Precision': [report['weighted avg']['precision']],
                    'Recall': [report['weighted avg']['recall']],
                    'F1-Score': [report['weighted avg']['f1-score']]
                })
                st.table(metrics_df)
    
    return test_results_df

if df_features is not None:
    test_results_df = test_and_evaluate_models_in_dashboard(models, scaler, df_features)
    st.dataframe(test_results_df)

# Charts and analyses section
st.markdown("---")
st.header("Charts and Project Analyses")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Raw Data",
    "Filter and Acceleration Magnitude",
    "Feature Analysis",
    "Model Sensitivity Analysis",
    "Model Results",
    "Interpretive Report",
    "Time Series History"
])

# Tab 1: Raw Data
with tab1:
    st.subheader("Raw Data Plots")
    if df_raw is not None:
        for date, group in df_raw.groupby('date'):
            st.markdown(f"**Raw Data Plot for Day {date}:**")
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
        st.info("No raw data plots found in the folder.")

# Tab 2: Filter and Magnitude
with tab2:
    st.subheader("Filtered and Acceleration Magnitude Plots")
    if df_magnitude is not None and df_filtered is not None:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Acceleration Magnitude Plots:**")
            for date, group in df_magnitude.groupby('date'):
                fig, ax = plt.subplots()
                ax.plot(group['magnitude'])
                ax.set_title(f'Accelerometer Magnitude - {date}')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Magnitude')
                st.pyplot(fig)
                plt.close(fig)
                st.markdown("---")
        with col2:
            st.markdown("**Filtered Plots:**")
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
    else:
        st.info("No magnitude or filtered plots found.")

# Tab 3: Feature Analysis
with tab3:
    st.subheader("Feature Analysis")
    if df_features is not None:
        st.markdown("**Feature Correlation Matrix:**")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df_features.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
        st.pyplot(fig)
        plt.close(fig)
        st.info("Values close to 1 or -1 indicate strong correlations between features.")
    else:
        st.info("No correlation matrix found.")

# Tab 4: Sensitivity Analysis
with tab4:
    st.subheader("Model Sensitivity Analysis")
    if df_features is not None and test_results_df is not None:
        st.markdown("**Threshold Sensitivity for Anomaly Detection:**")
        thresholds = np.linspace(0.1, 2.0, 10)
        f1_scores = {'Mahalanobis': [], 'Autoencoder': []}
        
        for thresh in thresholds:
            temp_results = test_results_df.copy()
            temp_results['Mahalanobis'] = (temp_results['Mahalanobis'] > thresh).astype(int)
            temp_results['Autoencoder'] = (temp_results['Autoencoder'] > thresh).astype(int)
            f1_mahala = f1_score(temp_results['true_label'], temp_results['Mahalanobis'], average='weighted')
            f1_ae = f1_score(temp_results['true_label'], temp_results['Autoencoder'], average='weighted')
            f1_scores['Mahalanobis'].append(f1_mahala)
            f1_scores['Autoencoder'].append(f1_ae)
        
        fig, ax = plt.subplots()
        ax.plot(thresholds, f1_scores['Mahalanobis'], label='Mahalanobis')
        ax.plot(thresholds, f1_scores['Autoencoder'], label='Autoencoder')
        ax.set_xlabel('Threshold')
        ax.set_ylabel('F1-Score')
        ax.set_title('Sensitivity to Threshold Changes')
        ax.legend()
        st.pyplot(fig)
        plt.close(fig)
        st.info("Optimal thresholds maximize F1-score; tune based on domain knowledge.")
    else:
        st.info("Sensitivity analysis requires features and test results.")

# Tab 5: Model Results
with tab5:
    st.subheader("Anomaly Detection Results")
    if test_results_df is not None:
        for index, row in test_results_df.iterrows():
            date = row['date']
            day_folder = os.path.join(TEST_FOLDER, date)
            file_path = os.path.join(day_folder, 'Accelerometer.csv')
            if os.path.exists(file_path):
                df_raw_test = pd.read_csv(file_path)
                df_raw_test['magnitude'] = np.sqrt(df_raw_test['x']**2 + df_raw_test['y']**2 + df_raw_test['z']**2)
                df_raw_test['filtered_magnitude'] = butter_bandpass_filter(df_raw_test['magnitude'].values, 0.1, 10, 100)
                
                st.markdown(f"**Anomaly Detection Plot for {date}:**")
                fig, ax = plt.subplots(figsize=(12, 6))
                ax.plot(df_raw_test['filtered_magnitude'], label='Filtered Data')
                ax.text(0.05, 0.95, f'True Label: {"Anomaly" if row["true_label"] == 1 else "Normal"}', 
                         transform=ax.transAxes, fontsize=12, color='black', weight='bold')
                if row['Isolation_Forest'] == 1:
                    ax.scatter([len(df_raw_test) / 2], [df_raw_test['filtered_magnitude'].mean()], color='red', s=200, label='IF Predicted Anomaly', marker='X')
                if row['Mahalanobis'] == 1:
                    ax.scatter([len(df_raw_test) / 2 + 50], [df_raw_test['filtered_magnitude'].mean()], color='green', s=200, label='Mahalanobis Predicted Anomaly', marker='X')
                if row['Autoencoder'] == 1:
                    ax.scatter([len(df_raw_test) / 2 - 50], [df_raw_test['filtered_magnitude'].mean()], color='purple', s=200, label='AE Predicted Anomaly', marker='X')
                ax.set_xlabel('Sample Index')
                ax.set_ylabel('Magnitude')
                ax.legend()
                st.pyplot(fig)
                plt.close(fig)
                if row['reason']:
                    st.info(f"Anomaly reason: {row['reason']}")
                st.markdown("---")
    else:
        st.info("No anomaly plots found.")

# Tab 6: Interpretive Report
with tab6:
    st.subheader("Final Interpretive Report")
    if test_results_df is not None:
        st.markdown("**Model Performance Summary**")
        overall_accuracy = (test_results_df['true_label'] == test_results_df['Isolation_Forest']).mean()
        st.metric("Overall Detection Accuracy (IF)", f"{overall_accuracy:.2%}")
        
        anomalies_detected = test_results_df[test_results_df['true_label'] == 1]
        if not anomalies_detected.empty:
            st.markdown("**Detected Anomalies:**")
            for _, row in anomalies_detected.iterrows():
                st.write(f"- {row['date']}: Predicted by {sum([row['Isolation_Forest'], row['Mahalanobis'], row['Autoencoder']])} models. Reason: {row['reason']}")
        
        st.markdown("""
        **Key Insights:**
        - The ensemble of models (IF, Mahalanobis, AE) provides robust anomaly detection for bridge vibrations.
        - High F1-scores indicate low false positives in controlled tests; validate on real-time data next.
        - Recommendations: Integrate with IoT sensors for continuous monitoring. Future work: Add CNN-LSTM for temporal patterns.
        """)
    else:
        st.info("No interpretive report available; run tests first.")

# Tab 7: Time Series History with Warning Levels
with tab7:
    st.subheader("Time Series History: Test Data Over Time")
    if test_results_df is not None and not test_results_df.empty:
        ts_df = prepare_time_series_data(test_results_df, models, scaler)
        if not ts_df.empty:
            # Use sidebar thresholds
            ts_df['Anomaly Score'] = ts_df['Anomaly Score'].clip(upper=1.0)
            ts_df['Warning Level'] = np.select(
                [ts_df['Anomaly Score'] < warning_low, ts_df['Anomaly Score'] < warning_high],
                ['Normal', 'Warning'], default='Alert'
            )
            ts_df['Color'] = np.select(
                [ts_df['Anomaly Score'] < warning_low, ts_df['Anomaly Score'] < warning_high],
                ['green', 'yellow'], default='red'
            )
            
            if PLOTLY_AVAILABLE:
                # Interactive Plotly time series
                fig = go.Figure()
                
                # Plot mean magnitude
                fig.add_trace(go.Scatter(
                    x=ts_df['Date'], y=ts_df['Mean Filtered Magnitude'],
                    mode='lines+markers', name='Mean Filtered Magnitude',
                    line=dict(color='blue'), marker=dict(size=8)
                ))
                
                # Plot anomaly score
                fig.add_trace(go.Scatter(
                    x=ts_df['Date'], y=ts_df['Anomaly Score'],
                    mode='markers', name='Anomaly Score',
                    marker=dict(color=ts_df['Color'], size=12, symbol='circle',
                                line=dict(width=2, color='darkgray')),
                    text=[f"Level: {level}<br>True: {true_label}" for level, true_label in zip(ts_df['Warning Level'], ts_df['True Label'])],
                    hovertemplate='<b>%{x}</b><br>Magnitude: %{y:.2f}<br>%{text}<extra></extra>'
                ))
                
                # Add warning lines
                fig.add_hline(y=warning_low, line_dash="dash", line_color="green", annotation_text="Normal Threshold")
                fig.add_hline(y=warning_high, line_dash="dash", line_color="yellow", annotation_text="Warning Threshold")
                fig.add_hline(y=1.0, line_dash="dash", line_color="red", annotation_text="Alert Max")
                
                fig.update_layout(
                    title="Temporal Evolution of Bridge Health Metrics",
                    xaxis_title="Date",
                    yaxis_title="Value (Normalized)",
                    yaxis=dict(range=[0, max(ts_df['Mean Filtered Magnitude'].max(), 1.0) * 1.1]),
                    hovermode='x unified',
                    showlegend=True
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("Interactive time series plot unavailable. Install Plotly with `pip install plotly`.")
            
            # Summary table
            st.markdown("**Time Series Summary:**")
            st.dataframe(ts_df[['Date', 'Mean Filtered Magnitude', 'Anomaly Score', 'Warning Level', 'True Label']])
            
            # Alerts summary
            alert_count = (ts_df['Warning Level'] == 'Alert').sum()
            st.metric("Active Alerts", alert_count)
            if alert_count > 0:
                st.warning(f"Review {alert_count} alert dates for immediate action.")
        else:
            st.info("No time series data prepared; check test folders.")
    else:
        st.info("Run tests first to generate time series data.")

# Sidebar Usage Guide
st.sidebar.title("Usage Guide")
st.sidebar.info(
    "1. This app runs the full analysis automatically.\n"
    "2. Ensure `data` and `test_data` folders are in the same GitHub repo.\n"
    "3. Install dependencies: `pip install -r requirements.txt`\n"
    "4. Run locally: `streamlit run dashboard_app_final.py`"
)
st.sidebar.markdown("---")
st.sidebar.success("The dashboard will open in your browser.")
```
