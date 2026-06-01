KAGGLE_DATASET = "kutayahin/wikipedia-link-graph-100k"

# Paths
DATA_DIR = "data"
LINKS_CSV = "data/links_export.csv"
PAGES_CSV = "data/pages_export.csv"
ARTICLES_JSON = "data/articles.json"
GRAPH_SCORES_JSON = "data/graph_scores.json"
TOP_PAGES_JSON = "data/top_pages.json"
EXAMPLES_JSON = "data/examples.json"
MODEL_PATH = "data/best_model.pt"

# Graph
GRAPH_ROWS = 20_000        # number of samples from links_export.csv
PAGERANK_ALPHA = 0.85
PAGERANK_MAX_ITER = 50
TOP_PAGES_COUNT = 50

# Wikipedia API
ARTICLES_COUNT = 100        # number of articles
ARTICLE_MAX_CHARS = 5000    # max number of simbols per article
API_SLEEP_SEC = 0.1         # pause

# Model
INPUT_DIM = 4               # tfidf, pagerank, position, length
HIDDEN_DIM_1 = 32
HIDDEN_DIM_2 = 16
DROPOUT = 0.3
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
EPOCHS = 20
VAL_SPLIT = 0.2

# Summary
SUMMARY_SENTENCES = 3        # number of sentences in summary
SUMMARY_EXAMPLES = 5         # number of articles at the end