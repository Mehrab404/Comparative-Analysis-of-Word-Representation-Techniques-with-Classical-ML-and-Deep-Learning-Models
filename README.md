## README

# Word Representations for Multiclass Text Classification
A comparative study of classical ML and deep learning models paired with Bag-of-Words, TF-IDF, GloVe, and Skip-gram embeddings for multiclass text classification across ten balanced categories.
Key result: Skip-gram embeddings with a Deep Neural Network (DNN) deliver the strongest overall performance, achieving 0.679 accuracy and 0.672 macro F1 on the held-out test set.

## Overview
This project evaluates how different word representation methods interact with classical and neural models for multiclass text classification, with a large, balanced dataset and rigorous preprocessing, training, and evaluation protocols.
Representations: BoW, TF-IDF, GloVe (100d), and task-trained Skip-gram (300d); Models: Logistic Regression, Naive Bayes, Random Forest, DNN, RNN, GRU, LSTM, and bidirectional variants.
The study demonstrates consistent gains from neural embeddings over sparse features, with Skip-gram + DNN outperforming all baselines and recurrent variants on accuracy and F1 metrics.

## Files
- Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification.pdf - full report with methods, experiments, figures, and tables.
- Model_Codes.ipynb - end-to-end notebook for preprocessing, feature extraction, model training, and evaluation.
- Project_Plots.zip - contains all the performance metrics plots of the models

## Key findings
- Skip-gram + DNN: Accuracy 0.679, Macro-F1 0.672, Weighted-F1 0.672 (best overall).
- GloVe + DNN: Accuracy 0.662, strongest among pre-trained embedding setups in this study.
- TF-IDF + Logistic Regression: Accuracy 0.637, a solid classical baseline that improves over BoW.
- SimpleRNN variants underperform due to vanishing gradients; GRU/LSTM show moderate gains but trail the DNN on this task.

## Dataset
The training set contains 279,999 samples and the test set 59,999 samples, with near-uniform distribution across ten topical classes for fair multiclass evaluation.
Texts mostly range from 30–120 words and 200–700 characters with a long tail, reflecting typical Zipfian length patterns in language data.
No missing values were detected, simplifying preprocessing and eliminating imputation steps.

## Preprocessing
- Lowercasing, punctuation/special character removal, whitespace normalization, tokenization.
- Stopword removal and lemmatization (WordNet) to reduce noise while preserving syntactic validity.
- Label encoding; for deep models, Keras Tokenizer with vocabulary size 10,000 and sequence length 100 via padding/truncation.

## Representations
- Bag-of-Words: Sparse term-count vectors via CountVectorizer for baselines.[1]
- TF-IDF: Unigram+bigram TfidfVectorizer to downweight ubiquitous terms and enrich local context.
- GloVe: 100-dimensional pre-trained vectors (Wikipedia + Gigaword) to initialize an embedding layer; OOV words use random vectors.
- Skip-gram: Task-specific 300-dimensional embeddings trained with Gensim on the project’s corpus, capturing domain semantics effectively.

## Models
- Classical: Logistic Regression, Multinomial Naive Bayes, Random Forest for sparse features and baselines.
- Deep: DNN (ReLU + softmax, dropout), SimpleRNN, GRU, LSTM, and bidirectional counterparts via Keras/TensorFlow with Adam and categorical cross-entropy.
- Training: Early stopping for regularization; all deep models operate on sequences up to 100 tokens with representation-aligned embedding sizes.

## Results
- BoW: DNN (0.623 acc) > Logistic Regression (0.612) ≈ Naive Bayes (0.602); Random Forest underperforms on high-dimensional sparse inputs.
- TF-IDF: Improves consistently over BoW; Logistic Regression at 0.637 acc; DNN and Naive Bayes also gain.
- GloVe: DNN reaches 0.662 acc; GRU 0.642; LSTM 0.630; SimpleRNN struggles due to vanishing gradients.
- Skip-gram: DNN leads overall with 0.679 acc and 0.672 F1 scores; recurrent variants trail but generally improve over GloVe counterparts.

## Environment
- Stack: Python with TensorFlow/Keras, scikit-learn, Gensim, and NLTK to match preprocessing and modeling pipelines.
- NLTK resources: WordNet lemmatizer and English stopwords required for preprocessing and should be downloaded in setup.
- Settings align with typical Python NLP workflows to support straightforward reproduction.

## Installation
Create and activate a Python environment (venv or conda), then install required packages and NLTK corpora before running preprocessing and training.

```
pip install tensorflow keras scikit-learn gensim nltk
python -c "import nltk; nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('omw-1.4')"
```

## Configuration
Default experimental settings: vocab_size = 10000, max_seq_len = 100, GloVe_dim = 100, Skipgram_dim = 300, optimizer = Adam, loss = categorical_crossentropy, early_stopping = True.
The dataset is balanced across 10 classes; initial subsampling of 50% of training data was used to manage resources without compromising representativeness.

Example YAML:
```
experiment:
  vocab_size: 10000
  max_seq_len: 100
  embedding:
    type: [bow, tfidf, glove, skipgram]
    glove_dim: 100
    skipgram_dim: 300
  model:
    type: [logreg, nb, rf, dnn, rnn, gru, lstm, bi_gru, bi_lstm]
  training:
    optimizer: adam
    loss: categorical_crossentropy
    early_stopping: true
    batch_size: 64
    epochs: 20
```

## Reproduction guide
- Preprocess the dataset with the provided pipeline to ensure identical tokenization, stopword removal, and lemmatization.
- Generate representations: BoW/TF-IDF via scikit-learn, GloVe-initialized embedding matrices, and Skip-gram embeddings trained with Gensim on the corpus.
- Train and evaluate each model–embedding combination, reporting accuracy, macro F1, weighted F1, confusion matrices, and classification reports.

## Training Skip-gram embeddings (example)
The study trains Skip-gram embeddings in-domain using Gensim to capture task-specific semantics with 300-dimensional vectors.

```
from gensim.models import Word2Vec
# tokens: list[list[str]] from the preprocessed corpus
w2v = Word2Vec(
    sentences=tokens,
    vector_size=300,  # per paper
    sg=1,             # Skip-gram
    workers=4,
    window=5,
    min_count=2,
    epochs=5
)
# Export embedding matrix aligned to tokenizer word_index
```

## Running models (examples)
- Classical: Train Logistic Regression, Naive Bayes, and Random Forest on BoW/TF-IDF features and evaluate on the test set.
- Deep: Build Keras models for DNN/RNN/GRU/LSTM (and bidirectional variants), initialize embeddings (GloVe or Skip-gram), and train with early stopping.

DNN sketch:
```
import tensorflow as tf
from tensorflow.keras import layers, models
model = models.Sequential([
  layers.Embedding(input_dim=vocab_size, output_dim=emb_dim, input_length=max_len, weights=[embedding_matrix], trainable=True),
  layers.GlobalAveragePooling1D(),
  layers.Dense(256, activation='relu'),
  layers.Dropout(0.5),
  layers.Dense(num_classes, activation='softmax')
])
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

## Evaluation
- Report Accuracy, Macro F1, and Weighted F1, alongside confusion matrices and classification reports for thorough multiclass assessment.
- Highlight the best configuration: Skip-gram + DNN (Accuracy 0.679; Macro-F1 0.672; Weighted-F1 0.672) and discuss trade-offs for TF-IDF + Logistic Regression and GloVe + DNN.

## Limitations and headroom
- Recurrent models are computationally expensive and susceptible to vanishing gradients in SimpleRNN variants on longer sequences.
- Processing and resource constraints required subsampling, fixed sequence lengths (100 tokens), and a capped vocabulary (10,000), with overall accuracy plateauing near 68% under these settings.
- With fewer processing constraints—longer sequences, larger vocabularies, full-data training, broader hyperparameter sweeps, and heavier architectures—overall performance would likely increase beyond current levels.

## Future work
- Evaluate transformer-based architectures (e.g., BERT, DistilBERT) for contextual embeddings and attention-driven gains.
- Explore data augmentation, targeted hyperparameter tuning (embedding dimensions, hidden units, dropout), and model ensembling to push results further.

## Project structure (suggested)
This structure supports clarity and reproducibility; adapt paths and names to the existing codebase if it differs.

```
.
├── data/
├── notebooks/
│   └── Model Codes.ipynb
├── reports/
│   └── Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification.pdf
├── src/
│   ├── preprocessing/
│   ├── features/  # bow, tfidf, glove_init, skipgram_train
│   ├── models/    # classical.py, dnn.py, rnn.py, gru.py, lstm.py
│   └── eval/
├── configs/
├── results/
└── README.md
```

## Citation
If using this project in academic or industrial work, cite: “Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification” by Mehrabul Islam and Md. Rumman Shahriar (BRAC University).
See the "Comparative Analysis of Word Representation Techniques with Classical ML and Deep Learning Models for Multiclass Text Classification" report for full methodology, dataset details, preprocessing, architectures, and comparative results with figures and tables.

## Acknowledgments
- Libraries: TensorFlow/Keras, scikit-learn, Gensim, and NLTK for modeling and preprocessing pipelines.
- Embeddings: GloVe for pre-trained vectors and Gensim Word2Vec (Skip-gram) for task-specific embeddings driving the strongest results.

