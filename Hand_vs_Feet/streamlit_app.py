import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import json
import matplotlib.pyplot as plt
import seaborn as sns
import mne
import os
import tempfile
from mne.datasets import eegbci
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd

st.set_page_config(
    page_title="EEG Model Evaluation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 EEG Model Evaluation: Confusion Matrix")
st.markdown("""
Aplikasi ini mengevaluasi performa model pada **seluruh event** yang ada dalam file EDF.
Ia membandingkan **Prediksi Model** vs **Label Asli (Marker T1/T2)**.
""")

@st.cache_resource
def load_assets():
    try:
        import os
        # 1. Dapatkan lokasi absolut file script ini (streamlit_app.py) berada
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 2. Sambungkan lokasi folder tersebut dengan nama file model
        model_path = os.path.join(base_dir, 'best_model.keras')
        mapping_path = os.path.join(base_dir, 'label_mapping.json')
        scaler_path = os.path.join(base_dir, 'scaler_raw.pkl')

        # 3. Load menggunakan path lengkap (absolut) tadi
        model = tf.keras.models.load_model(model_path)
        
        with open(mapping_path, 'r') as f:
            mapping = json.load(f)
            
        scaler = joblib.load(scaler_path)
        
        return model, mapping, scaler
        
    except Exception as e:
        st.error(f"Terjadi error saat load assets: {e}")
        return None, None, None

model, mapping_info, scaler_info = load_assets()

if mapping_info is None:
    mapping_info = {
        'input_shape': [64, 881],
        'sampling_rate': 160,
        'class_mapping': {'0': 'Hands (Tangan)', '1': 'Feet (Kaki)'}
    }
    scaler_info = {'type': 'none'}

def process_edf_batch(file_path, target_fs=160):
    """
    Mengambil SEMUA epoch T1 dan T2 dari file untuk evaluasi batch.
    Returns: 
        X (numpy array): Data sinyal [n_epochs, 64, timepoints]
        y (numpy array): Label asli [n_epochs] (0 atau 1)
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
        
        mne.datasets.eegbci.standardize(raw)
        try:
            raw.set_montage('standard_1020')
        except:
            pass 

        if raw.info['sfreq'] != target_fs:
            raw.resample(target_fs, npad='auto')

        if len(raw.ch_names) > 64:
            raw.pick(raw.ch_names[:64])
        elif len(raw.ch_names) < 64:
            st.error(f"Channel kurang dari 64 ({len(raw.ch_names)}).")
            return None, None

        raw.filter(7., 60., fir_design='firwin', skip_by_annotation='edge', verbose=False)

        events, event_id = mne.events_from_annotations(raw, verbose=False)
                
        needed_events = ['T1', 'T2']
        found_events = {k: v for k, v in event_id.items() if k in needed_events}
        
        if not found_events:
            st.error("Tidak ada marker T1/T2.")
            return None, None

        tmin, tmax = 0.5, 6.0
        epochs = mne.Epochs(raw, events, event_id=found_events, 
                            tmin=tmin, tmax=tmax, baseline=None, preload=True, verbose=False)
        
        X = epochs.get_data() 
        
        y_mne_ids = epochs.events[:, -1]
        
        id_map = {}
        if 'T1' in found_events: id_map[found_events['T1']] = 0
        if 'T2' in found_events: id_map[found_events['T2']] = 1
        
        y = np.array([id_map[i] for i in y_mne_ids])
        
        return X, y

    except Exception as e:
        st.error(f"Error processing: {e}")
        return None, None

st.sidebar.header("📂 Sumber Data Evaluasi")
input_source = st.sidebar.radio("Pilih Sumber:", ("MNE Dataset (Online)", "Upload File (.edf)"))

file_path = None
temp_file = None

if input_source == "Upload File (.edf)":
    uploaded = st.sidebar.file_uploader("Upload .edf", type=['edf'])
    if uploaded:
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".edf")
        temp_file.write(uploaded.getvalue())
        temp_file.close()
        file_path = temp_file.name

else:
    st.sidebar.info("Dataset: EEGBCI (Motor Execution)")
    subject_id = st.sidebar.number_input("Subject ID (1-109)", 1, 109, 1)
    
    run_id = st.sidebar.selectbox("Run (Task 4: Hands vs Feet)", [5, 9, 13])
    
    if st.sidebar.button("Load Data"):
        with st.spinner("Downloading..."):
            paths = eegbci.load_data(subject_id, [run_id], update_path=True, verbose=False)
            file_path = paths[0]
            st.sidebar.success(f"File loaded: {os.path.basename(file_path)}")

if file_path and model:
    if st.button("🚀 Jalankan Evaluasi Full", type="primary"):
        with st.spinner("Memproses seluruh epoch & memprediksi..."):
            
            X, y_true = process_edf_batch(file_path)
            
            if X is not None:
                if scaler_info.get('type') == 'manual_multiplier':
                    X_processed = X * scaler_info['factor']
                else:
                    X_processed = X
                
                X_input = X_processed.reshape(X.shape[0], 64, 881, 1)
                
                y_probs = model.predict(X_input, verbose=0)
                y_pred = (y_probs > 0.5).astype(int).flatten()
                
                acc = accuracy_score(y_true, y_pred)
                
                col_metrics, col_matrix = st.columns([1, 2])
                
                with col_metrics:
                    st.subheader("📋 Ringkasan")
                    st.metric("Total Epochs", len(y_true))
                    st.metric("Akurasi", f"{acc*100:.2f}%")
                    
                    st.markdown("---")
                    st.text("Detail Laporan:")
                    report = classification_report(y_true, y_pred, target_names=['Hands', 'Feet'], output_dict=True)
                    st.dataframe(pd.DataFrame(report).transpose().style.format("{:.2f}"))

                with col_matrix:
                    st.subheader("🟦 Confusion Matrix")
                    
                    cm = confusion_matrix(y_true, y_pred)
                    
                    fig, ax = plt.subplots(figsize=(6, 5))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                                xticklabels=['Pred: Hands', 'Pred: Feet'],
                                yticklabels=['True: Hands', 'True: Feet'], ax=ax)
                    
                    ax.set_ylabel('Label Asli (Ground Truth)', fontsize=12)
                    ax.set_xlabel('Prediksi Model', fontsize=12)
                    st.pyplot(fig)
                
                st.subheader("🔍 Analisis Error")
                if acc < 1.0:
                    st.write("Epoch yang salah diprediksi:")
                    incorrect_indices = np.where(y_true != y_pred)[0]
                    
                    idx_err = incorrect_indices[0]
                    true_label_str = "Hands" if y_true[idx_err] == 0 else "Feet"
                    pred_label_str = "Hands" if y_pred[idx_err] == 0 else "Feet"
                    
                    st.warning(f"Epoch ke-{idx_err} | Asli: **{true_label_str}** | Prediksi: **{pred_label_str}**")
                    
                    fig_err, ax_err = plt.subplots(figsize=(10, 3))
                    ax_err.plot(X[idx_err, 0, :], label='Ch 1') 
                    ax_err.plot(X[idx_err, 10, :], label='Ch 10')
                    ax_err.set_title(f"Sinyal Epoch {idx_err} (Misclassified)")
                    st.pyplot(fig_err)
                else:
                    st.success("Sempurna! Semua epoch diprediksi dengan benar.")

    if temp_file:
        os.remove(file_path)

elif not file_path:
    st.info("👈 Silakan load data dari sidebar untuk memulai evaluasi.")
elif not model:
    st.error("Model tidak ditemukan.")