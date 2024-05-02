# Telco-RAG: Retrieval-Augmented Generation for Telecommunications

Telco-RAG is a specialized Retrieval-Augmented Generation (RAG) framework designed to tackle the unique challenges presented by the telecommunications industry, particularly when working with complex and rapidly evolving 3GPP documents.

## References
- Bornea, A.-L., Ayed, F., De Domenico, A., Piovesan, N., & Maatouk, A. (2024). Telco-RAG: Navigating the Challenges of Retrieval-Augmented Language Models for Telecommunications. *arXiv preprint arXiv:2404.15939*. [DOI](https://doi.org/10.48550/arXiv.2404.15939) | [Read the paper](https://arxiv.org/pdf/2404.15939.pdf)


## Features

- **Custom RAG Pipeline**: Tailored specifically to handle the intricacies of telecommunications standards.
- **Enhanced Query Processing**: Utilizes a dual-stage query enhancement and retrieval process to improve the accuracy and relevance of generated responses.
- **Hyperparameter Optimization**: Carefully tuned to deliver the best performance by optimizing chunk sizes, context length, and embedding models.
- **NN Router**: A neural network-based router that improves the efficiency and accuracy of document retrieval, significantly reducing RAM usage.
- **Open-Source**: Freely available for the community to use, adapt, and improve upon.

## Getting Started

To get started with Telco-RAG, you'll need to clone the repository and set up the environment:

```bash
git clone https://github.com/netop-team/telco-rag.git
cd <repository-directory>
```
Prerequisites
Python 3.8+
Uvicorn
Other dependencies listed in requirements.txt

Installation
Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Running the API
To launch the API server, use the following command:
```bash
uvicorn main:app --reload
```

License
This project is licensed under the MIT License - see the LICENSE file for details.
