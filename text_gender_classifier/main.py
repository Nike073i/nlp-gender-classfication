from .rf_classifier import load_classifier
import fire
import os
import sys

MODEL_PATH = './models/rf_url.pkl'

def check_file(file_path):
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f"Error: File '{file_path}' does not exist or is not a file", file=sys.stderr)
        sys.exit(1)

def read_text(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            return content
    except UnicodeDecodeError:
        print(f"Error: File '{file_path}' is not UTF-8 encoded.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"Error reading file: {str(e)}", file=sys.stderr)
        sys.exit(1)
    
def classify_gender(text_path):
    check_file(text_path)
    text = read_text(text_path)
    
    predict = load_classifier(MODEL_PATH)
    gender = predict(text)[0]

    output = f'Its woman' if gender == 0 else 'Its man'
    print(output)
    

if __name__ == '__main__':
    fire.Fire(classify_gender)