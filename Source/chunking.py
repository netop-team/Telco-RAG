import re

def custom_text_splitter(text, chunk_size, chunk_overlap, word_split=False):
    """
    Splits a given text into chunks of a specified size with a defined overlap between them.

    This function divides the input text into chunks based on the specified chunk size and overlap.
    Optionally, it can split the text at word boundaries to avoid breaking words when 'word_split'
    is set to True. This is achieved by using a regular expression that identifies word separators.

    Args:
        text (str): The text to be split into chunks.
        chunk_size (int): The size of each chunk in characters.
        chunk_overlap (int): The number of characters of overlap between consecutive chunks.
        word_split (bool, optional): If True, ensures that chunks end at word boundaries. Defaults to False.

    Returns:
        list of str: A list containing the text chunks.
    """
    chunks = []
    start = 0
    separators_pattern = re.compile(r'[\s,.\-!?\[\]\(\){}":;<>]+')
    
    while start < len(text) - chunk_overlap:
        end = min(start + chunk_size, len(text))
        
        if word_split:
            match = separators_pattern.search(text, end)
            if match:
                end = match.end()
                
        if end == start:
            end = start + 1
            
        chunks.append(text[start:end])
        start = end - chunk_overlap
        
        if word_split:
            match = separators_pattern.search(text, start-1)
            if match:
                start = match.start() + 1
                
        if start < 0:
            start = 0
    
    return chunks


def chunk_doc(doc):
    chunks= custom_text_splitter( doc["text"], 500, 25, word_split = True)
    return [{"text": chunk, "source": doc["source"]} for chunk in chunks]
 
