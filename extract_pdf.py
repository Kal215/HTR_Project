import sys
import subprocess

try:
    import PyPDF2
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "PyPDF2"])
    import PyPDF2

reader = PyPDF2.PdfReader('D:/dokumentasi_skripsi/HASIL_DOKUMENTASI/PDF/13.UNIKOM_MUHAMMAD RISKAL FADHILLA_BAB 3.pdf')
text = ""
for page in reader.pages:
    text += page.extract_text() + "\n"

print(text)
