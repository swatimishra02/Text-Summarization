# Converting pdf to text when pdf is saved on the system. This pdf raw text will also be saved in 'pdf_text.txt'

import PyPDF2

# function to extract text from a pdf
def extract_text_from_pdf(file_path):
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        text = ''
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            text += page.extract_text()
        return text


# enter the path of the pdf saved on the system. This can further be improved by taking pdf/document/text as input.
file_path = r"C:\Users\KIIT\Downloads\Test Files\Patent Document\US9038026.pdf"
pdf_text = extract_text_from_pdf(file_path)

# writing the extracted text in the 'pdf_text.txt' file.
with open('pdf_text.txt', 'w', encoding="utf-8") as f:
    f.write(pdf_text)

print(pdf_text)

#-----------------------------------------------------------------------------------------------
# Code for saving the pdf text in a csv file.
# data = []
#for i in pdf_text:
 #   name = i
  #  data.add(name)
#print(data)

#df = pd.DataFrame({"Data": data})

#df.to_csv("patent_data.csv")


