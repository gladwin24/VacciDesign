import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import time

# Load trained XGBoost model
model = joblib.load("D:\\iisc_pro\\VacciDes\\xgb_best_model.pkl")

# One-hot encoding function for 21-mer peptides
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
aa_to_index = {aa: idx for idx, aa in enumerate(amino_acids)}

def one_hot_encode(peptide, max_len=21):
    matrix = np.zeros((max_len, 21), dtype=np.float32)
    for i, aa in enumerate(peptide[:max_len]):
        if aa in aa_to_index:
            matrix[i, aa_to_index[aa]] = 1
    return matrix.flatten()

def predict_peptides(peptides):
    encoded = np.array([one_hot_encode(p) for p in peptides])
    probs = model.predict_proba(encoded)[:, 1]
    return probs

def generate_vaccine(peptides, threshold=0.5, progress_callback=None, time_callback=None):
    scores = []
    total = len(peptides)
    start_time = time.time()
    for idx, p in enumerate(peptides):
        encoded = np.array([one_hot_encode(p)])
        prob = model.predict_proba(encoded)[:, 1][0]
        scores.append(prob)
        if progress_callback:
            progress_callback((idx + 1) / total)
        if time_callback:
            elapsed = time.time() - start_time
            est_total = (elapsed / (idx + 1)) * total
            remaining = est_total - elapsed
            time_callback(remaining)

    high_conf_peptides = [p for p, score in zip(peptides, scores) if score >= threshold]
    high_conf_peptides = list(dict.fromkeys(high_conf_peptides))  # remove duplicates
    if not high_conf_peptides:
        return "", [], scores

    adjuvant = "AKFVAAWTLKAAA"  # PADRE
    linker = "AAY"
    final_sequence = adjuvant + linker + linker.join(high_conf_peptides)
    return final_sequence, high_conf_peptides, scores

def show_prediction_plot(peptides, scores, threshold):
    df_plot = pd.DataFrame({"Peptide": peptides, "Score": scores}).sort_values("Score", ascending=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    colors = ['green' if s >= threshold else 'red' for s in df_plot["Score"]]
    ax.bar(df_plot["Peptide"], df_plot["Score"], color=colors)
    ax.axhline(threshold, color="blue", linestyle="--", label=f"Threshold = {threshold}")
    ax.set_title("Prediction Scores")
    ax.set_ylabel("Score")
    ax.set_xticklabels(df_plot["Peptide"], rotation=90)
    ax.legend()
    st.pyplot(fig)

# --- Streamlit UI ---
st.title("VacciDes")
st.markdown("Generate multi-epitope vaccine using XGBoost predictions")

uploaded_file = st.file_uploader("Upload neoantigen peptide file (TSV)", type=["tsv"])
threshold = st.slider("Prediction threshold", min_value=0.0, max_value=1.0, value=0.5, step=0.01)

if uploaded_file:
    df = pd.read_csv(uploaded_file, sep="\t")
    st.write("### Uploaded Data", df.head())

    if "Mutant_Peptide" not in df.columns:
        st.error("The uploaded file must contain a 'Mutant_Peptide' column.")
    else:
        peptides = df["Mutant_Peptide"].dropna().unique().tolist()
        if st.button("Generate Vaccine"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            time_text = st.empty()

            def update_progress(p):
                percent = int(p * 100)
                progress_bar.progress(percent)
                status_text.text(f"Processing: {percent}%")

            def update_time(remaining):
                time_text.text(f"Estimated time remaining: {int(remaining)} seconds")

            sequence, selected_peptides, scores = generate_vaccine(
                peptides,
                threshold=threshold,
                progress_callback=update_progress,
                time_callback=update_time
            )

            progress_bar.empty()
            status_text.empty()
            time_text.empty()
            show_prediction_plot(peptides, scores, threshold)

            if sequence:
                st.success(f"Generated vaccine sequence with {len(selected_peptides)} epitopes.")
                st.download_button("Download FASTA", data=f">vaccine\n{sequence}", file_name="vaccine.fasta")

                csv_data = pd.DataFrame({"Selected_Peptides": selected_peptides})
                st.download_button("Download Epitope CSV", data=csv_data.to_csv(index=False), file_name="selected_peptides.csv")

                st.text_area("Vaccine Sequence", sequence, height=200)
                st.markdown("You can paste this sequence into [ColabFold](https://colab.research.google.com/github/sokrypton/ColabFold/blob/main/AlphaFold2_mmseqs2.ipynb) for structure prediction.")
            else:
                st.warning("No peptides passed the prediction threshold.")
