import sys
import subprocess

# Try to install pdfminer.six if not available
try:
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams
except ImportError:
    print("Installing pdfminer.six...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pdfminer.six"])
    from pdfminer.high_level import extract_text
    from pdfminer.layout import LAParams

pdf_path = r"d:\TU HOC\geometric-ml-foundations\docs\109600_5E4Hp1V.pdf"
output_path = r"d:\TU HOC\geometric-ml-foundations\docs\slides.txt"

laparams = LAParams(
    line_margin=0.5,
    word_margin=0.1,
    char_margin=2.0,
    all_texts=True
)

print(f"Extracting text from: {pdf_path}")
text = extract_text(pdf_path, laparams=laparams)

with open(output_path, "w", encoding="utf-8") as f:
    f.write(text)

print(f"Done! Saved to: {output_path}")
print(f"Total characters extracted: {len(text)}")
print("\n--- PREVIEW (first 3000 chars) ---\n")
print(text[:3000])
