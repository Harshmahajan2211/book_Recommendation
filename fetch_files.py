import gdown
import os

def download_files():
    if not os.path.exists("data"):
        os.makedirs("data")

    file_ids = {
        "Books.csv": "1EeMLA8mBSLTXJcszy6JLv1qFnGUY89UP",
        "Ratings.csv": "10rOVJ7t2-nMCjBixo3Uiw4qABhC3vMol",
        "Users.csv": "1wwN3NmAvdWBK9ILY9F68KRjty8UbZC0o"
    }

    for filename, file_id in file_ids.items():
        output_path = os.path.join("data", filename)
        if not os.path.exists(output_path):
            gdown.download(f"https://drive.google.com/uc?id={file_id}", output_path, quiet=False)

