# Telco-RAG: Retrieval-Augmented Generation for Telecommunications

Telco-RAG is a specialized Retrieval-Augmented Generation (RAG) framework designed to tackle the unique challenges presented by the telecommunications industry, particularly when working with complex and rapidly evolving 3GPP documents. Developed at the Paris Research Center by Huawei Technologies and in collaboration with Yale University, this framework enhances the capabilities of large language models by integrating retrieval mechanisms that provide context-specific support for generating accurate and relevant responses.

## Features

- **Custom RAG Pipeline**: Tailored specifically to handle the intricacies of telecommunications standards.
- **Enhanced Query Processing**: Utilizes a dual-stage query enhancement and retrieval process to improve the accuracy and relevance of generated responses.
- **Hyperparameter Optimization**: Carefully tuned to deliver the best performance by optimizing chunk sizes, context length, and embedding models.
- **NN Router**: A neural network-based router that improves the efficiency and accuracy of document retrieval, significantly reducing RAM usage.
- **Open-Source**: Freely available for the community to use, adapt, and improve upon.

## Getting Started

To get started with Telco-RAG, you'll need to clone the repository and set up the environment:

```bash
git clone <repository-url>
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
