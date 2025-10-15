import pandas as pd

FILE_PATH = "merged_liga1.xlsx"

# tampilkan semua sheet yang benar-benar ada di file
xls = pd.ExcelFile(FILE_PATH)
print("Daftar sheet yang ditemukan:", xls.sheet_names)
