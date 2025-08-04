import gdown
import os

def download_files():
    if not os.path.exists("data"):
        os.makedirs("data")

    file_ids = {
        "books.csv": "1EeMLA8mBSLTXJcszy6JLv1qFnGUY89UP",
        "ratings.csv": "10rOVJ7t2-nMCjBixo3Uiw4qABhC3vMol",
        "users.csv": "1wwN3NmAvdWBK9ILY9F68KRjty8UbZC0o"
    }

    for filename, file_id in file_ids.items():
        output_path = os.path.join("data", filename)
        gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

# Run this function once to fetch all three files
download_files()
