import logging
import numpy as np
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_embeddings(series_docs):
    """Add embeddings to each chunk of documents from pre-saved NumPy files."""
    for doc_key, doc_chunks in series_docs.items():
        try:
            # Load embeddings specific to each document series
            embeddings = np.load(f'3GPP-Release18\Embeddings\Embeddings{doc_key}.npy')
        except FileNotFoundError:
            logging.error(f"Embedding file for {doc_key} not found.")
            continue
        except Exception as e:
            logging.error(f"Failed to load embeddings for {doc_key}: {e}")
            continue

        # Process each chunk within the document series
        updated_chunks = []
        for chunk in doc_chunks:
            for idx, single_chunk in enumerate(chunk):
                try:
                    single_chunk['embedding'] = embeddings[idx]
                    updated_chunks.append(single_chunk)
                except IndexError:
                    logging.warning(f"Embedding index {idx} out of range for {doc_key}.")
                except Exception as e:
                    logging.error(f"Error processing chunk {idx} for {doc_key}: {e}")
        
        series_docs[doc_key] = updated_chunks

    return series_docs

