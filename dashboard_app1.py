# -*- coding: utf-8 -*-

import streamlit as st
import pandas as pd
import os
import glob
from PIL import Image
import datetime

# تنظیمات اصلی صفحه
st.set_page_config(
    page_title="داشبورد پایش سلامت سازه",
    layout="wide",
    initial_sidebar_state="expanded"
)

# مسیرهای ورودی و خروجی
OUTPUT_FOLDER = 'output'
RAW_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, 'raw_data')
MAGNITUDE_FOLDER = os.path.join(OUTPUT_FOLDER, 'acceleration_magnitude')
FILTERED_FOLDER = os.path.join(OUTPUT_FOLDER, 'filtered_data')
FEATURE_DATA_FOLDER = os.path.join(OUTPUT_FOLDER, 'feature_data')
FEATURE_ANALYSIS_FOLDER = os.path.join(OUTPUT_FOLDER, 'feature_analysis')
SENSITIVITY_ANALYSIS_FOLDER = os.path.join(OUTPUT_FOLDER, 'sensitivity_analysis')
TEST_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'test_results')
TRAINING_RESULTS_FOLDER = os.path.join(OUTPUT_FOLDER, 'training_results')

# توابع کمکی
def load_image(file_path):
    try:
        return Image.open(file_path)
    except FileNotFoundError:
        st.warning(f"فایل تصویری یافت نشد: {file_path}")
        return None

# عنوان اصلی
st.title("داشبورد پایش سلامت سازه پل کابلی تبریز")
st.markdown("---")

# -----------------------------------------------------------
# ۱. نمایش وضعیت تحلیل
# -----------------------------------------------------------
st.header("خلاصه وضعیت تحلیل")

# نمایش نتایج تست
try:
    df_results = pd.read_csv(os.path.join(TEST_RESULTS_FOLDER, 'test_results.csv'))
    st.markdown("**نتایج تشخیص ناهنجاری روی داده‌های جدید:**")
    st.dataframe(df_results)
except FileNotFoundError:
    st.info("فایل نتایج تست یافت نشد. لطفاً ابتدا کد تحلیل را اجرا کنید.")

# نمایش گزارش متنی
try:
    st.markdown("---")
    st.markdown("**گزارش ارزیابی مدل‌ها:**")
    with open(os.path.join(TEST_RESULTS_FOLDER, 'evaluation_metrics.txt'), 'r', encoding='utf-8') as f:
        metrics_report = f.read()
    st.text(metrics_report)
except FileNotFoundError:
    st.info("گزارش ارزیابی مدل‌ها یافت نشد.")

# -----------------------------------------------------------
# ۲. بخش نمایش نمودارها و تحلیل‌ها
# -----------------------------------------------------------
st.markdown("---")
st.header("نمایش نمودارها و تحلیل‌های پروژه")

# ایجاد تب‌های مختلف
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "داده‌های خام",
    "فیلتر و دامنه شتاب",
    "تحلیل ویژگی‌ها",
    "تحلیل حساسیت",
    "نتایج مدل‌ها",
    "گزارش تفسیری",
    "پایش و هشدارها"  # تب جدید
])

# تب ۱: داده‌های خام
with tab1:
    st.subheader("نمودارهای داده‌های خام")
    raw_files = sorted(glob.glob(os.path.join(RAW_DATA_FOLDER, "*.png")))
    if raw_files:
        for file in raw_files:
            date = os.path.basename(file).split('_')[2].split('.')[0]
            st.markdown(f"**نمودار داده‌های خام برای روز {date}:**")
            st.image(file, use_column_width=True)
            st.markdown("---")
    else:
        st.info("هیچ نمودار داده خام در پوشه مربوطه یافت نشد.")

# تب ۲: فیلتر و دامنه شتاب
with tab2:
    st.subheader("نمودارهای فیلتر و دامنه شتاب")
    col1, col2 = st.columns(2)
    mag_files = sorted(glob.glob(os.path.join(MAGNITUDE_FOLDER, "*.png")))
    filter_files = sorted(glob.glob(os.path.join(FILTERED_FOLDER, "*.png")))
    
    if mag_files:
        with col1:
            st.markdown("**نمودارهای دامنه شتاب:**")
            for file in mag_files:
                date = os.path.basename(file).split('_')[1].split('.')[0]
                st.image(file, caption=f"دامنه شتاب - {date}", use_column_width=True)
                st.markdown("---")
    if filter_files:
        with col2:
            st.markdown("**نمودارهای فیلترشده:**")
            for file in filter_files:
                date = os.path.basename(file).split('_')[2].split('.')[0]
                st.image(file, caption=f"فیلتر باند-پاس - {date}", use_column_width=True)
                st.markdown("---")
    if not mag_files and not filter_files:
        st.info("هیچ نمودار دامنه یا فیلترشده‌ای یافت نشد.")

# تب ۳: تحلیل ویژگی‌ها
with tab3:
    st.subheader("تحلیل ویژگی‌ها")
    corr_matrix_file = os.path.join(FEATURE_ANALYSIS_FOLDER, 'correlation_matrix.png')
    if os.path.exists(corr_matrix_file):
        st.markdown("**ماتریس همبستگی ویژگی‌ها:**")
        st.image(corr_matrix_file, use_column_width=True)
        st.info("نزدیکی مقادیر به ۱ یا -۱ نشان‌دهنده همبستگی قوی بین ویژگی‌هاست.")
    else:
        st.info("ماتریس همبستگی یافت نشد.")

# تب ۴: تحلیل حساسیت
with tab4:
    st.subheader("تحلیل حساسیت مدل‌ها")
    sensitivity_plot_file = os.path.join(SENSITIVITY_ANALYSIS_FOLDER, 'sensitivity_analysis.png')
    if os.path.exists(sensitivity_plot_file):
        st.markdown("**تحلیل حساسیت (تعداد ناهنجاری‌ها بر اساس آستانه):**")
        st.image(sensitivity_plot_file, use_column_width=True)
    else:
        st.info("نمودار تحلیل حساسیت یافت نشد.")
        
    try:
        sensitivity_report_file = os.path.join(SENSITIVITY_ANALYSIS_FOLDER, 'sensitivity_report.txt')
        with open(sensitivity_report_file, 'r', encoding='utf-8') as f:
            report_content = f.read()
        st.markdown("**گزارش متنی تحلیل حساسیت:**")
        st.text(report_content)
    except FileNotFoundError:
        st.info("گزارش متنی تحلیل حساسیت یافت نشد.")

# تب ۵: نتایج مدل‌ها
with tab5:
    st.subheader("نتایج تشخیص ناهنجاری")
    anomaly_files = sorted(glob.glob(os.path.join(TEST_RESULTS_FOLDER, "anomalies_*.png")))
    if anomaly_files:
        for file in anomaly_files:
            date = os.path.basename(file).split('_')[1].split('.')[0]
            st.markdown(f"**نمودار تشخیص ناهنجاری برای {date}:**")
            st.image(file, use_column_width=True)
            st.markdown("---")
    else:
        st.info("هیچ نمودار ناهنجاری‌ای در پوشه مربوطه یافت نشد.")

# تب ۶: گزارش تفسیری
with tab6:
    st.subheader("گزارش تفسیری نهایی")
    interpretation_file = os.path.join(TEST_RESULTS_FOLDER, 'interpretation_report.txt')
    if os.path.exists(interpretation_file):
        try:
            with open(interpretation_file, 'r', encoding='utf-8') as f:
                report_content = f.read()
            st.text(report_content)
        except Exception as e:
            st.error(f"خطا در خواندن گزارش تفسیری: {e}")
    else:
        st.info("گزارش تفسیری یافت نشد. لطفاً ابتدا کد تحلیل را اجرا کنید.")

# تب ۷: پایش و هشدارها (جدید)
with tab7:
    st.subheader("پایش بلادرنگ و سیستم هشدارهای ایمیلی")
    st.markdown("این بخش، وضعیت پایش لحظه‌ای و گزارش هشدارهای حیاتی سیستم را نمایش می‌دهد.")
    st.markdown("---")

    # نمایش آخرین زمان اجرای تحلیل
    try:
        last_run_time = os.path.getmtime(os.path.join(TEST_RESULTS_FOLDER, 'test_results.csv'))
        dt_object = datetime.datetime.fromtimestamp(last_run_time)
        st.info(f"آخرین زمان اجرای تحلیل داده‌ها: **{dt_object.strftime('%Y-%m-%d %H:%M:%S')}**")
    except FileNotFoundError:
        st.warning("فایل نتایج تست یافت نشد. نمی‌توان آخرین زمان اجرا را نمایش داد.")

    st.markdown("---")

    # نمایش هشدارهای حیاتی (بر اساس اجماع مدل‌ها)
    st.subheader("گزارش هشدارهای حیاتی (ارسال ایمیل) 🚨")
    try:
        df_results = pd.read_csv(os.path.join(TEST_RESULTS_FOLDER, 'test_results.csv'))
        
        # پیدا کردن ردیف‌هایی که هر سه مدل ناهنجاری را تأیید کرده‌اند
        critical_alerts = df_results[
            (df_results['Isolation_Forest'] == 1) &
            (df_results['Mahalanobis'] == 1) &
            (df_results['Autoencoder'] == 1)
        ]
        
        if not critical_alerts.empty:
            st.error("سیستم یک ناهنجاری **حیاتی** را شناسایی کرده و ایمیل هشدار ارسال شده است.")
            st.dataframe(critical_alerts[['date', 'reason']])
            st.markdown("---")
            st.info("**توضیح:** این جدول فقط روزهایی را نمایش می‌دهد که هر سه مدل به صورت همزمان ناهنجاری را تشخیص داده‌اند که منجر به ارسال ایمیل شده است.")
        else:
            st.success("هیچ هشدار حیاتی (بر اساس اجماع سه مدل) در داده‌ها یافت نشد. وضعیت پایش عادی است.")
    except FileNotFoundError:
        st.info("فایل نتایج تست برای نمایش هشدارها یافت نشد.")

# -----------------------------------------------------------
# راهنمای استفاده
# -----------------------------------------------------------
st.sidebar.title("راهنمای استفاده")
st.sidebar.info(
    "1. ابتدا فایل `SHM2.py` را اجرا کنید تا تمام داده‌ها، نمودارها و گزارش‌ها تولید شوند.\n"
    "2. سپس، ترمینال را در پوشه پروژه خود باز کنید و دستور زیر را اجرا کنید:\n\n"
    "`streamlit run dashboard_app.py`"
)
st.sidebar.markdown("---")
st.sidebar.success("با این کار، داشبورد در مرورگر شما باز خواهد شد.")
