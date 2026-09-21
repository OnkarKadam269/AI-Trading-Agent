import PyPDF2

reader = PyPDF2.PdfReader(r'C:\Users\Omkar\OneDrive\Desktop\AI AGENT\05784-AAAI26.ShiY-ML.pdf')

with open(r'C:\Users\Omkar\OneDrive\Desktop\AI AGENT\backend\kronos_paper.txt', 'w', encoding='utf-8') as f:
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        f.write(f"\n--- PAGE {i+1} ---\n")
        f.write(text)
        
print(f"Extracted {len(reader.pages)} pages successfully.")
