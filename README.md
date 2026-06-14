# BROpt —

This report presents the work of our group on multi-label movie genre classification from short text descriptions. The
task is to assign one or more genres to each movie - it's quite a tricky problem, since genres are subjective,
descriptions are short, and the label distribution is heavily imbalanced.
#RUN final_version.py in backend

### We built and compared three different systems:

**TF-IDF + MLP** - a classic baseline

**FastText + TF-IDF** weighted MLP - something in between

**DistilBERT** - a pretrained transformer, fine-tuned on our data

#### **The DistilBERT model showed the best results:**

Micro F1 of 0.66, Precision of 0.76, Recall of 0.70.

# Dataset
We used IMDb-based dataset across our three models, since we started with one and switched to another partway through the project.
All models use the IMDb Movie Genre Classification dataset in TMDB format. It consists of two CSV files: movies_overview.csv, which contains movie descriptions and numeric genre IDs, and movies_genres.csv, which maps those IDs to human-readable genre names. Genre IDs are stored as JSON-like strings (e.g. "[18, 80]") and need to be parsed and mapped at load time.


# Preprocessing

**TF-IDF MLP:** lowercase + strip. That's basically it, TF-IDF handles the rest internally.

**FastText MLP:** lowercase + remove all punctuation with regex. Important because FastText looks words up in a
dictionary - punctuation attached to a word breaks the lookup.

**DistilBERT**: no manual cleaning needed - the tokenizer handles everything.

**All models**: 70/15/15 train/val/test split, fixed seed 42. MultiLabelBinarizer fits only on training data.

# Model Type

**TF-IDF + MLP** (baseline)
**TF-IDF** converts each description into a sparse vector of 20,000 features (unigrams + bigrams), weighted by how
discriminative each term is. On top of that, a 3-layer MLP: 20000 -> 512 -> 256 -> 128 -> K. BatchNorm and Dropout(0.3)
between layers. Quite simple and reproducible.

FastText + TF-IDF weighted MLP

Instead of TF-IDF features, we use pretrained FastText embeddings (Common Crawl, 300 dimensions). For each description,
we average the word vectors, but weighted by TF-IDF IDF scores - so rare, genre-specific words like "heist" or "
dystopian" matter more than common words.

The 300D vector goes into an MLP with residual connections: two blocks of Linear -> BatchNorm -> ReLU, with a skip
connection around the first block. Residual connections help gradients flow during training.

DistilBERT
DistilBERT is a smaller, faster version of BERT - 40% fewer parameters, 60% faster, but keeps about 97% of the
performance. We fine-tune it on our data with a small classification head on top: Linear(768->256) -> ReLU -> Dropout(
0.3) -> Linear(256->K).

# Methodology Overview

### All three models follow the same overall rules:

* Load and clean data
* Convert text to vectors (TF-IDF / FastText / tokenizer)
* Train with early stopping on validation Micro F1
* Search for the best threshold on the validation set
* Evaluate on the held-out test set

# Input and Output

## Input

* TF-IDF MLP: sparse vector, 20,000 dimensions
* FastText MLP: dense vector, 300 dimensions (L2-normalized)
* DistilBERT: sequence of up to 128 token IDs + attention mask

# Output
# Evaluation Metrics

We use four metrics:

* Micro F1 - main metric. Aggregates TP/FP/FN across all genres before computing F1. In other words it counts all
  correctly guessed genres
* Precision - what fraction of predicted genre assignments are correct.
* Recall - what fraction of true genres we actually caught.
* Exact Match Accuracy - actually if we predicted Strictest metric, naturally low.
* Lose -

# Results

* TF-IDF + MLP:
    * micro f1: 0.55
    * lose: 0.13
* FastText + MLP:
    * micro f1: 0.6
    * lose: 0.23
* DistilBERT:
    * micro f1: 0.66
    * lose: 0.015
