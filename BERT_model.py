import numpy as np
import re
import torch
from scipy.sparse.linalg import svds
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import BertTokenizer
from nltk.corpus import stopwords
from string import punctuation
import PyPDF2
from rouge_score import rouge_scorer

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Example usage to extract text from a PDF file
file_path = r"C:\Users\KIIT\Downloads\Test Files\Patent Document\US9038026.pdf"
pdf_text = extract_text_from_pdf(file_path)

# Data cleaning

DOCUMENT = str(pdf_text)

# Remove newline and carriage return characters
DOCUMENT = re.sub(r'\n|\r', ' ', DOCUMENT)
# Replace multiple spaces with a single space
DOCUMENT = re.sub(r' +', ' ', DOCUMENT)
# Remove leading and trailing spaces
DOCUMENT = DOCUMENT.strip()

# Tokenize the document into sentences
sentences = sent_tokenize(DOCUMENT)

# Set stop words to English
stop_words = set(stopwords.words('english') + list(punctuation))

# Function to normalize and clean a document
def normalize_document(doc):
    doc = re.sub(r'[^a-zA-Z\s]', '', doc, re.I|re.A)  # Remove special characters
    doc = doc.lower()  # Convert to lowercase
    doc = doc.strip()  # Remove leading and trailing spaces
    tokens = word_tokenize(doc)  # Tokenize document
    filtered_tokens = [token for token in tokens if token not in stop_words]  # Remove stopwords
    doc = ' '.join(filtered_tokens)  # Recreate document from filtered tokens
    return doc

# Vectorize the normalization function
normalize_corpus = np.vectorize(normalize_document)

# Normalize each sentence in the document
norm_sentences = normalize_corpus(sentences)

# TF-IDF keyword extraction

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer(min_df=0., max_df=1., ngram_range=(1, 3), use_idf=True)
# Fit and transform the normalized sentences
tfidf_matrix = tfidf_vectorizer.fit_transform(norm_sentences)
# Convert the TF-IDF matrix to a dense array
tfidf_matrix = tfidf_matrix.toarray()

# Get feature names from the TF-IDF vectorizer
feature_names = tfidf_vectorizer.get_feature_names_out()
# Transpose the TF-IDF matrix for further processing
tfidf_matrix_df = tfidf_matrix.T

# BERT encoder

# Initialize BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Add special tokens [CLS] and [SEP]
marked_doc = "[CLS]" + DOCUMENT + "[SEP]"
# Tokenize the document
sent_embds = tokenizer.tokenize(marked_doc)
segments_ids = [1] * len(sent_embds)

# Map the token strings to their vocabulary indices
indexed_tokens = tokenizer.convert_tokens_to_ids(sent_embds)

# Convert inputs to PyTorch tensors
tokens_tensor = torch.tensor([indexed_tokens])
segments_tensors = torch.tensor([segments_ids])

# Function to perform low-rank SVD
def low_rank_svd(matrix, singular_count=2):
    u, s, vt = svds(matrix, k=singular_count)
    return u, s, vt

# Number of sentences and topics for SVD
num_sentences = 5
num_topics = 5

# Perform low-rank SVD on the TF-IDF matrix
u, s, vt = low_rank_svd(tfidf_matrix_df, singular_count=num_topics)

term_topic_mat, singular_values, topic_document_mat = u, s, vt

# Thresholding singular values
sv_threshold = 0.5
min_sigma_value = max(singular_values) * sv_threshold
singular_values[singular_values < min_sigma_value] = 0

# Calculate salience scores
salience_scores = np.sqrt(np.dot(np.square(singular_values), np.square(topic_document_mat)))

# Rouge evaluation

# Function to calculate Rouge scores
def rouge_evaluation(reference, hypothesis):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores

# Get top sentence indices based on salience scores
top_sentence_indices = (-salience_scores).argsort()[:num_sentences]
top_sentence_indices.sort()

# Create the generated summary
generated_summary = ' '.join([norm_sentences[index] for index in top_sentence_indices])

# Reference summary (you may replace it with a manually written summary for evaluation)
with open('reference_summary.txt', 'r', encoding="utf-8") as f:
    reference_summary = f.read()

# Rouge evaluation
scores = rouge_evaluation(reference_summary, generated_summary)

# Print Rouge scores
print("ROUGE-1 Precision:", scores['rouge1'].precision)
print("ROUGE-1 Recall:", scores['rouge1'].recall)
print("ROUGE-1 F1 Score:", scores['rouge1'].fmeasure)

print("ROUGE-2 Precision:", scores['rouge2'].precision)
print("ROUGE-2 Recall:", scores['rouge2'].recall)
print("ROUGE-2 F1 Score:", scores['rouge2'].fmeasure)

print("ROUGE-L Precision:", scores['rougeL'].precision)
print("ROUGE-L Recall:", scores['rougeL'].recall)
print("ROUGE-L F1 Score:", scores['rougeL'].fmeasure)
