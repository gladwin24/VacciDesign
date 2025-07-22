# VacciDesign
AI-Powered Personalized Cancer Vaccine Design Platform

# AI-Powered Personalized Cancer Vaccine Design Platform

![Project Banner](https://img.shields.io/badge/AI--Cancer--Vaccine-Neoantigen--Design-blue)

## 📌 Overview

This project is an end-to-end AI-powered platform for designing personalized cancer vaccines by analyzing tumor biopsy-derived DNA and RNA sequencing data. It identifies immunogenic neoantigens using machine learning, ranks them based on predicted immune response potential, and generates codon-optimized mRNA vaccine sequences ready for synthesis.

---

## 🎯 Objectives

* Predict patient-specific neoantigens from tumor mutations.
* Use AI models to assess HLA binding and immunogenicity.
* Generate codon-optimized vaccine sequences.
* Provide a user-friendly interface for clinicians/researchers.

---

## 🧠 Technologies Used

### 💻 Programming & AI

* Python 3.10
* PyTorch
* XGBoost
* Scikit-learn

### 🧬 Bioinformatics Tools
* TCGA Dataset
* OptiType (HLA typing)
* Ensembl VEP (mutation annotation)
* Biopython

### 📊 Data Handling & Visualization

* Pandas, NumPy
* Matplotlib, Plotly

### 🌐 Web App Interface

* Streamlit (Primary UI)
* Flask (Optional backend)

### 🚀 Deployment

* Google Colab Pro / AWS EC2
* Streamlit Cloud / Render

---

## 🧪 Features

* Upload `.maf` and RNA `.tsv` files
* Accept patient-specific HLA alleles
* Predict, filter, and rank neoantigens
* Codon-optimize peptides for vaccine output
* Export results as `.csv` and `.fasta`

---

## 🏗️ System Architecture

```
[User Interface (Streamlit)]
         ↓
[Preprocessing Module (.maf/.tsv/HLA)]
         ↓
[Neoantigen Prediction Engine (AI models + NetMHCpan)]
         ↓
[Codon Optimization & Scoring]
         ↓
[Downloadable Vaccine Sequences (.csv/.fasta)]
```

---

## 🧰 Installation

```bash
# Clone the repository
$ git clone https://github.com/gladwin24/VacciDesign
$ cd VacciDesign

# Create virtual environment
$ python -m venv venv
$ source venv/bin/activate  # on Windows use `venv\Scripts\activate`

# Install dependencies
$ pip install -r requirements.txt

# Run the Streamlit app
$ streamlit run app.py
```

---

## 📂 Project Structure

```
├── app.py                  # Streamlit UI
├── pipeline/               # Core ML and bioinformatics logic
│   ├── predictor.py        # AI models for HLA binding & immunogenicity
│   ├── optimizer.py        # Codon optimization logic
│   └── utils.py            # Preprocessing tools
├── data/                   # Sample inputs (MAF, RNA, HLA)
├── output/                 # Generated vaccine sequences
├── requirements.txt        # Python dependencies
└── README.md               # Project overview
```

---

## 🧪 Sample Files

* `data/sample.maf`
* `data/expression.tsv`
* `data/hla_typing.tsv`

---

## 💡 Future Enhancements

* Integrate AlphaFold for structural validation
* Add support for additional cancer types
* Improve prediction models with CRISPR-screen training data
* FDA-ready data pipeline for clinical trials

---

## 🧑‍💻 Contributors

* Gladwin Alappat
* Joel Joshy

---

## 📃 License

MIT License – free to use for academic and research purposes.

---

## 📬 Contact

For queries, collaborations, or demos, please contact:
📧 [gladwin14@gmail.com](mailto:gladwin14@gmail.com)
🌐 [GitHub](https://github.com/gladwin24)

---

> Empowering faster, smarter, and accessible cancer immunotherapy through AI 💉
