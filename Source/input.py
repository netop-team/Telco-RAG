import os
from tqdm.auto import tqdm
from doc2docx import convert
from docx import Document
from Source.storage import Storage
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def read_docx(file_path):
    """Read and extract text from a DOCX file."""
    try:
        doc = Document(file_path)
        return '\n'.join(para.text for para in doc.paragraphs)
    except Exception as e:
        logging.error(f"Failed to read DOCX file at {file_path}: {e}")
        return None

def get_documents(series_list, folder_path=r'Documents', storage_name='Documents.db', dataset_name="Standard"):
    """Retrieve and process documents from a folder, storing them in a database if not already present."""
    storage = Storage(f'.\Database\{storage_name}')
    storage.create_dataset(dataset_name)
    
    document_ds = []
    file_list = []

    # Check and convert .doc files to .docx
    convert_docs_to_docx(folder_path)

    # Prepare list of .docx files for processing
    file_list = [f for f in tqdm(os.listdir(folder_path), desc="Filtering documents") if valid_file(f, series_list)]

    # Process each document
    for filename in tqdm(file_list, desc="Processing documents"):
        file_path = os.path.join(folder_path, filename)
        process_document(file_path, filename, storage, document_ds, dataset_name)

    storage.close()
    return document_ds

def convert_docs_to_docx(folder_path):
    """Convert .doc files in a folder to .docx format if any."""
    has_doc = any(f.endswith('.doc') for f in os.listdir(folder_path))
    if has_doc:
        convert(folder_path)

def valid_file(filename, series_list):
    """Check if a file should be processed based on its name and series list."""
    return filename.endswith(".docx") and not filename.startswith("~$") and (not filename[:2].isnumeric() or int(filename[:2]) in series_list)

def process_document(file_path, filename, storage, document_ds, dataset_name):
    """Process a single document file."""
    if storage.is_id_in_dataset(dataset_name, filename):
        data_dict = storage.get_dict_by_id(dataset_name, filename)
        document_ds.append(data_dict)
    else:
        content = read_docx(file_path)
        if content:
            data_dict = {'id': filename, 'text': content, 'source': filename}
            document_ds.append(data_dict)
            storage.insert_dict(dataset_name, data_dict)

