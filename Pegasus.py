from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import PyPDF2
from rouge_score import rouge_scorer

# Load tokenizer and model for Pegasus
tokenizer = PegasusTokenizer.from_pretrained("google/pegasus-xsum")
model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum")

# Function to extract text from a PDF file
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        # Use PyPDF2 to read the PDF file
        reader = PyPDF2.PdfReader(file)
        text = ''
        # Iterate through each page in the PDF and extract text
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text

# Function to generate a summary using Pegasus model
def generate_summary(text):
    # Tokenize the input text and prepare it for model input
    tokens = tokenizer(text, truncation=True, padding="longest", return_tensors="pt")
    # Generate a summary using the Pegasus model
    summary = model.generate(**tokens)
    # Decode the generated summary tokens into human-readable text
    decoded_summary = tokenizer.decode(summary[0], skip_special_tokens=True)
    return decoded_summary

# Function to evaluate ROUGE scores between a reference and generated summary
def rouge_evaluation(reference, hypothesis):
    # Create a RougeScorer object with specified metrics
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    # Calculate ROUGE scores between the reference and generated summary
    scores = scorer.score(reference, hypothesis)
    return scores

file_path = r"C:\Users\KIIT\Downloads\Test Files\Patent Document\US9038026.pdf"
pdf_text = extract_text_from_pdf(file_path)

# Generate summary
generated_summary = generate_summary(pdf_text)

# Reference summary
with open('reference_summary.txt', 'r', encoding="utf-8") as f:
    reference_summary = f.read()

# Evaluate ROUGE scores between the reference and generated summary
scores = rouge_evaluation(reference_summary, generated_summary)

# Print ROUGE scores for precision, recall, and F1 score for different n-grams (1-gram, 2-gram, and Longest Common Subsequence)
print("ROUGE-1 Precision:", scores['rouge1'].precision)
print("ROUGE-1 Recall:", scores['rouge1'].recall)
print("ROUGE-1 F1 Score:", scores['rouge1'].fmeasure)

print("ROUGE-2 Precision:", scores['rouge2'].precision)
print("ROUGE-2 Recall:", scores['rouge2'].recall)
print("ROUGE-2 F1 Score:", scores['rouge2'].fmeasure)

print("ROUGE-L Precision:", scores['rougeL'].precision)
print("ROUGE-L Recall:", scores['rougeL'].recall)
print("ROUGE-L F1 Score:", scores['rougeL'].fmeasure)
