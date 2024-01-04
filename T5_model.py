import PyPDF2
from transformers import T5Tokenizer, T5ForConditionalGeneration
from rouge_score import rouge_scorer
import pandas as pd

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

# Function for abstractive summarization using T5 model
def abstractive_summarization(text):
    # Load pre-trained model and tokenizer
    model_name = "t5-small"
    tokenizer = T5Tokenizer.from_pretrained(model_name)
    model = T5ForConditionalGeneration.from_pretrained(model_name)

    # Tokenize input text
    inputs = tokenizer.encode("summarize: " + text, return_tensors="pt", max_length=512, truncation=True)

    # Generate summary
    summary_ids = model.generate(inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

    return summary

# Example usage for abstractive summarization
generated_summary = abstractive_summarization(pdf_text)
print("Generated Summary:\n", generated_summary)

# Save the generated summary to a file for reference
with open('generated_summary.txt', 'w', encoding="utf-8") as f:
    f.write(generated_summary)

# Read the reference summary from a file (you need to provide your own reference summary)
with open('reference_summary.txt', 'r', encoding="utf-8") as f:
    reference_summary = f.read()

# Compute Rouge scores
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
scores = scorer.score(generated_summary, reference_summary)

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
