# ArguSense Upgrade v3.0: Dual TF-IDF & Full-Data Argument Framework

ArguSense v3.0 is a significant upgrade to our NLP pipeline. We improved performance further using smarter feature engineering, full-dataset training, and focused hyperparameter tuning.

## 🚀 What's New in v3.0?
- **Dual TF-IDF Pipeline**: We now combine word-level TF-IDF with character-level TF-IDF (`char_wb`). This captures both semantic meaning and intra-word syntactic patterns, making sarcasm detection far more robust to misspellings and slang without adding blindly complex deep learning models.
- **Full-Dataset Argument Training**: By default, the system no longer heavily downsamples the argument dataset. `config.ARGUMENT_SAMPLE_SIZE` is now `None`, unlocking the full scale of the feedback dataset to drastically improve the capability of recognizing subtle discourse traits.
- **Improved Context Representation**: Combining `Discourse Type` labels directly with text (`Lead [SEP] When people say...`) ensures the models natively process the structural context.
- **Targeted Hyperparameter Tuning**: Rather than stacking arbitrary weak learners like PassiveAggressive classifiers, we exclusively use `GridSearchCV` on our highest-performing baseline models (e.g., Logistic Regression, Linear SVC, and SGD). 
- **Optional Feature Engineering**: Handcrafted linguistic features (counts, averages, exclamation ratios) can be dynamically toggled via `config.py`.

## 📂 Structural Upgrade
- `config.py`: Centralized configuration for `USE_WORD_TFIDF`, `USE_CHAR_TFIDF`, `ARGUMENT_SAMPLE_SIZE`, and parameter grids.
- `src/feature_engineering.py`: Refactored to generate and dynamically stack dual sparse vectorizers using `scipy.sparse.hstack`.
- `src/train_*.py`: Modernized to tune only the strongest models and persist the entire `ArguSenseFeaturePipeline`.
- `src/predict.py`: Refactored to seamlessly load and handle the multi-vectorizer pipeline at inference.
- `src/evaluate.py`: Added explicit tracking for `macro_f1` and `weighted_f1` for multiclass stability.

## 🛠️ Execution Guide

> [!IMPORTANT]
> Because models, vectorizers, and the complete preprocessing logic have been reconstructed, **you must execute the pipelines manually in VS Code** to generate the new artifacts before running the app.

### 1. Update Environment (Optional)
Ensure your environment has the latest prerequisites:
```bash
pip install -r requirements.txt
```

### 2. Retrain Sarcasm Pipeline
Tune and train the sarcasm detection system:
```bash
python src/train_sarcasm.py
```
*Outputs: Optimized `best_sarcasm_model.joblib` and corresponding `feature_pipeline.joblib`.*

### 3. Retrain Argument Pipeline
Train the full-scale argument evaluation models safely (Warning: Full dataset will take longer but yields higher precision):
```bash
python src/train_argument.py
```
*Outputs: `best_argument_model.joblib` and `feature_pipeline.joblib` alongside updated comparative metrics.*

### 4. Launch the Dashboard
Serve the local analytics dashboard utilizing the locally built models:
```bash
streamlit run app.py
```

## 🔬 Research & Implementation Methodology

This section outlines the scientific methodology and engineering architecture behind ArguSense, specifically designed to provide context for academic analysis and research posters.

### 1. System Architecture
ArguSense employs a **Dual-Pipeline Parallel Architecture** governed by a final Heuristic Fusion Layer:
- **Pipeline A (Sarcasm Detection):** A binary classification system designed to detect linguistic irony and sarcastic markers in text.
- **Pipeline B (Argument Effectiveness):** A multiclass stratification system designed to grade the rhetorical quality of text as *Ineffective*, *Adequate*, or *Effective*.
- **Sarcasm-Aware Fusion Layer:** A unique deterministic layer that intercepts the predictions from both pipelines. It re-evaluates argument quality through the lens of sarcasm, concluding whether the text utilizes "Effective Irony" to bolster a point, or if the sarcasm acts as a "Poor Sarcastic Effort" that critically undermines literal meaning.

### 2. Preprocessing Pipeline
Text ingestion applies strict standardization to limit noise while meticulously preserving semantic markers essential for classical ML models:
- **Discourse Combination:** For argument analysis, structural context (e.g., *Lead, Claim, Evidence*) is explicitly prepended to the text using a `[SEP]` delimiter (e.g., `Claim [SEP] The data shows...`). This explicitly teaches linear models to associate rhetorical structural tags with upcoming linguistic patterns.
- **Sarcasm Marker Preservation:** Unlike standard NLP pipelines that aggressively strip punctuation, ArguSense dynamically preserves exclamation marks (`!`) and question marks (`?`) to retain emotional velocity, which are paramount indicators for irony detection.
- **Normalization:** URLs, HTML artifacts, zero-width characters, and social media handles (`@mentions`) are completely stripped to regularize the vocabulary bounds.

### 3. Feature Engineering (Dual TF-IDF)
Rather than relying on resource-intensive deep learning embeddings (e.g., BERT), arguably the most significant innovation in ArguSense v3.0 is the **High-Dimensional Dual Sparse Vectorization**:
- **Semantic Word TF-IDF:** Captures broader syntactic relationships using a unigram and bigram (`1, 2`) scope capped at 50,000 features. We utilize `sublinear_tf=True` (logarithmic term frequency scaling) to strictly penalize repetitive keyword spamming and normalize document length distributions.
- **Syntactic Character TF-IDF (`char_wb`):** Operates entirely inside specific word boundaries (`char_wb`) using tri-grams to penta-grams (`3, 5`) capped at 30,000 features. This mathematically captures prefixes, suffixes, misspellings, and elongated slang (e.g., "soooo", "yaaas") which are historically the strongest indicators of modern digital sarcasm.
- **Orthogonal Modularity:** These vectorizers uniquely operate in tandem; they are asynchronously fitted and uniformly merged along the feature axis using `scipy.sparse.hstack`, generating a singular 80,000-dimensional matrix passed to the linear estimators.
- **Linguistic Handcrafted Features (Optional):** We generate lightweight analytical stats (uppercase ratio, punctuation density, average word length) yielding a quantifiable vector of shouting intensity, which is sequentially merged into the TF-IDF matrix.

### 4. Training Pipeline & Model Selection
The training suite utilizes strict, objective benchmarking built entirely on Scikit-Learn pipelines:
- **Targeted Sweep Configuration:** Rather than blind ensembles, we deploy rigorous localized `GridSearchCV` hyperparameter sweeps across three highly performant classical margin bounds: Regularized Logistic Regression, Linear Support Vector Classification (LinearSVC), and Online Stochastic Gradient Descent (SGD Log/Hinge loss).
- **Algorithmic Calibration:** Margin classifiers (like SVM) do not natively produce probabilistic thresholds. ArguSense programmatically handles distance-to-margin interpretations natively at inference, ensuring seamless confidence intervals without the mathematical decay of nested `CalibratedClassifierCV` wrapper fits.
- **Data Balancing & Verification:** The datasets are heavily skewed towards generalized claims. To prevent majoritarian prediction failures, we inject synthetic sample weighting uniformly across the parameter grid (`class_weight='balanced'`). The final architectures are strictly evaluated against `weighted_f1` and `macro_f1` scores to ensure minority-class robustness.
