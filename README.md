# NLU_Project

data_parser.py = script to parse the raw transcripts

Label_Scraping.ipynb = script used to scrape the labels from YahooFinance API

CountVectorizer Transcript Level.ipynb = implementation of count vectorizer based BoW benchmarks - with transcript level aggregation

CountVectorizer Chunk Level.ipynb = implementation of count vectorizer based BoW benchmarks

TFIDF Transcript.ipynb = implementation of TFIDF based BoW benchmarks - with transcript level aggregation

TFIDF Transcript.ipynb = implementation of TFIDF based BoW benchmarks

glue.py = where we modify the data processors of the transformers library

init.py = where we modify the metrics


The four sbatch files are used to run BERT large and ALBERT on NYU Prince. SCR = regression task, SCC = classification task

The two .out files are training and evaluation logs from slurm output. We only upload the classification output because the regression output files are larger than 25MB and GitHub prevents us from uploading them.
