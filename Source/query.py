import os
import json
from tqdm.auto import tqdm
from Source.retrieval import find_nearest_neighbors_faiss
from Source.index import get_faiss_batch_index
import openai
import chardet
from Source.get_definitions import define_TA_question
import numpy as np
import ast
import torch
from tqdm.auto import tqdm
import openai
from torch.utils.data import DataLoader, TensorDataset
import torch
import torch
import torch.nn as nn
import torch.nn.functional as F
import traceback

class CustomModel(nn.Module):
    def __init__(self):
        super(CustomModel, self).__init__()
        self.layer1_1 = nn.Linear(1024, 768)
        self.layer1_2 = nn.Linear(768, 512)
        self.layer1_3 = nn.Linear(512, 256)
        self.dropout1 = nn.Dropout(0.2)  

        self.layer2_1 = nn.Linear(18, 128)
        self.layer2_2 = nn.Linear(128, 256)
        self.dropout2 = nn.Dropout(0.05) 
        
        self.batchnorm1 = nn.BatchNorm1d(256)
        self.batchnorm2 = nn.BatchNorm1d(256)

        self.alfa = nn.Parameter(torch.ones(1), requires_grad=True)
        self.beta = nn.Parameter(torch.ones(1), requires_grad=True)

        self.output_layer1 = nn.Linear(256, 128)
        self.output_layer2 = nn.Linear(128, 18)
        
        self.leaky_relu = nn.LeakyReLU(0.01)

    def forward(self, input_1, input_2):

        x1 = F.relu(self.layer1_1(input_1))
        x1 = self.dropout1(x1)
        x1 = F.relu(self.layer1_2(x1))
        x1 = self.dropout1(x1)
        x1 = F.relu(self.layer1_3(x1))
        x1 = self.batchnorm1(x1)

        x2 = F.relu(self.layer2_1(input_2))
        x2 = self.dropout2(x2)
        x2 = F.relu(self.layer2_2(x2))
        x2 = self.batchnorm2(x2)

        weighted_x1 = self.alfa * x1
        weighted_x2 = self.beta * x2
        
        combined = weighted_x1 + weighted_x2
        
        output = self.output_layer1(self.leaky_relu(combined))
        output = self.output_layer2(self.leaky_relu(output))
        
        return output
    
    
class Query:
    def __init__(self, query, context):
        self.id = id
        self.question = query
        self.query = query 
        self.enhanced_query = query
        self.con_counter = {}
        self.topic_distr = []
        if isinstance(context, str):
            context = [context]
        self.context = context
        self.rowcontext = []
        self.context_source = []
        self.possible_sources = []
        self.wg = []
        self.source_hit={}
        self.document_accuracy = None
    

    def def_TA_question(self):
        self.query = define_TA_question(self.query)
        self.enhanced_query = self.query

    def candidate_answers(self):
            try:
                client = openai.OpenAI()
                try:
                    row_context = f"""
                    Provide all the possible answers to the fallowing question. Conisdering your knowledge and the text provided.
                    Question {self.query}\n
                    
                    Considering the fallowing context:
                    {self.context}
                    
                    Provide all the possible answers to the fallowing question. Conisdering your knowledge and the text provided.
                    Question {self.question}\n
                    
                    
                    Make sure none of the answers provided contradicts with your knowledge and have at most 100 characters each.
                    """
                    generated_output = client.chat.completions.create(
                    model = "gpt-3.5-turbo-1106",
                    messages = [
                        {"role": "system", "content": "You are an expert at telecom knowledge. Be concise, precise and provide exact technical terms required."},
                        {"role": "user", "content":  row_context},
                    
                    ],
                    )
                    generated_output_str = generated_output.choices[0].message.content
                    print(generated_output_str)
                    if generated_output_str != "NO": 
                        self.context = generated_output_str 
                        self.enhanced_query = self.query +'\n'+ self.context
                except Exception as e:
                    print(f"An error occurred: {e}")
            except:
                print("ERROR") 
                print(traceback.format_exc())   

    def get_embeddings_list(text_list):
        # Initialize the OpenAI client
        client = openai.OpenAI()
        # Request embeddings for the list of texts using a different, larger model
        response = client.embeddings.create(
                    input=text_list,
                    model="text-embedding-3-large",
                    dimensions=1024,
                )
        embeddings = []
        for i in range(len(response.data)):
            embeddings.append(response.data[i].embedding)
        
        text_embeddings= {}
        print(len(text_list))
        print(len(embeddings))
        for index in range(len(text_list)):
            text_embeddings[text_list[index]] = embeddings[index]
        return text_embeddings
    
    def inner_product(a, b):
        """Compute the inner product of two lists."""
        return sum(x * y for x, y in zip(a, b))
    
    def get_col2(embeddings_list):
        topics_with_series = [("Requirements (21 series): Focuses on the overarching requirements necessary for UMTS (Universal Mobile Telecommunications System) and later cellular standards, including GSM enhancements, security standards, and the general evolution of 3GPP systems. It covers vocabulary, security threats, UE capability requirements, and work items for various releases.", "21 series"),
            ("Service aspects ('stage 1') (22 series): This series details the initial specifications for services provided by the network, outlining the service requirements before the technical realization is detailed. It serves as the first step in defining what the network should provide.", "22 series"),
            ("Technical realization ('stage 2') (23 series): Focuses on the architectural and functional framework necessary to implement the services described in stage 1, providing a bridge to the detailed protocols and interfaces defined in stage 3​.,", "23 series"),
            ("Signalling protocols ('stage 3') - user equipment to network (24 series): Details the protocols and signaling procedures for communication between user equipment and the network, ensuring interoperability and successful service delivery.", "24 series"),
            ("Radio aspects (25 series): Covers the specifications related to radio transmission technologies, including frequency bands, modulation schemes, and antenna specifications, critical for ensuring efficient and effective wireless communication​.", "25 series"),
            ("CODECs (26 series): Contains specifications for voice, audio, and video codecs used in the network, defining how data is compressed and decompressed to enable efficient transmission over bandwidth-limited wireless networks.", "26 series"),
            ("Data (27 series): This series focuses on the data services and capabilities of the network, including specifications for data transmission rates, data service features, and support for various data applications.", "27 series"),
            ("Signalling protocols ('stage 3') - (RSS-CN) and OAM&P and Charging (overflow from 32.- range) (28 series): Addresses additional signaling protocols related to operation, administration, maintenance, provisioning, and charging, complementing the core signaling protocols outlined in the 24 series.", "28 series"),
            ("Signalling protocols ('stage 3') - intra-fixed-network (29 series): Specifies signaling protocols used within the fixed parts of the network, ensuring that various network elements can communicate effectively to provide seamless service to users.", "29 series"),
            ("Programme management (30 series): Relates to the management and coordination of 3GPP projects and work items, including documentation and specification management procedures​.", "30 series"),
            ("Subscriber Identity Module (SIM / USIM), IC Cards. Test specs. (31 series): Covers specifications for SIM and USIM cards, including physical characteristics, security features, and interaction with mobile devices, as well as testing specifications for these components​.", "31 series"),
            ("OAM&P and Charging (32 series): Focuses on operation, administration, maintenance, and provisioning aspects of the network, as well as the charging principles and mechanisms for billing and accounting of network services.", "32 series"),
            ("Security aspects (33 series): Details the security mechanisms and protocols necessary to protect network operations, user data, and communication privacy, including authentication, encryption, and integrity protection measures​.", "33 series"),
            ("UE and (U)SIM test specifications (34 series): Contains test specifications for User Equipment (UE) and (U)SIM cards, ensuring that devices and SIM cards meet 3GPP standards and perform correctly in the network​.", "34 series"),
            ("Security algorithms (35 series): Specifies the cryptographic algorithms used in the network for securing user data and signaling information, including encryption algorithms and key management procedures.", "35 series"),
            ("LTE (Evolved UTRA), LTE-Advanced, LTE-Advanced Pro radio technology (36 series): Details the technical specifications for LTE, LTE-Advanced, and LTE-Advanced Pro technologies, including radio access network (RAN) protocols, modulation schemes, and network architecture​.", "36 series"),
            ("Multiple radio access technology aspects (37 series): Addresses the integration and interoperability of multiple radio access technologies within the network, enabling seamless service across different types of network infrastructure.", "37 series"),
            ("Radio technology beyond LTE (38 series): Focuses on the development and specification of radio technologies that extend beyond the capabilities of LTE, aiming to improve speed, efficiency, and functionality for future cellular networks​.", "38 series")
        ]
        file_path = 'series_description.json'
        if os.path.isfile(file_path):
            # File exists, read the file and load the JSON content into series_dict
            with open(file_path, 'r') as file:
                series_dict = json.load(file)
        else:
            series_dict = {}
            for desc, series_index in topics_with_series:
                series_dict[series_index] = {}
                series_dict[series_index]["description"]= desc
                series_dict[series_index]["embeddings"]= Query.get_embeddings(desc)
            # File does not exist, write the series_dict to the file as JSON
            with open(file_path, 'w') as file:
                json.dump(series_dict, file, indent=4)
        
        similarity_coloumn = []
        for embeddings in embeddings_list:
            coef = []
            for series_id in series_dict:
                coef.append(Query.inner_product(embeddings, series_dict[series_id]['embeddings']))
            similarity_coloumn.append(coef)
        return similarity_coloumn
    
    def preprocessing_softmax(embeddings_list):
        embeddings = np.array(embeddings_list)
        similarity = np.array(Query.get_col2(embeddings))

        X_train_1_tensor = torch.tensor(embeddings, dtype=torch.float32)

        _similarity = torch.from_numpy(similarity)
        _similarity = torch.tensor(similarity, dtype=torch.float32)
        X_train_2_tensor= torch.nn.functional.softmax(10*_similarity, dim=-1)
    

        dataset = TensorDataset(X_train_1_tensor, X_train_2_tensor)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

        return dataloader
    
    def get_embeddings(text):
        # Initialize the OpenAI client
        client = openai.OpenAI()
        # Request embeddings for the list of texts using a different, larger model
        response = client.embeddings.create(
                    input=text,
                    model="text-embedding-3-large",
                    dimensions=1024,
                )
        return response.data[0].embedding

    def predict_wg(self):
        model = CustomModel()
        model.load_state_dict(torch.load('router_new.pth', map_location='cpu'))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()
        text_list = []
        text_embeddings = Query.get_embeddings_list([self.enhanced_query])
        label_list = []
        embeddings = text_embeddings[self.enhanced_query]
        test_dataloader= Query.preprocessing_softmax([embeddings])
        with torch.no_grad():
            for X1, X2 in test_dataloader:
                # Move data to the same device as the model
                X1, X2 = X1.to(device), X2.to(device)
                original_labels_mapping = np.arange(21, 39)
                outputs = model(X1, X2)
                top_values, top_indices = outputs.topk(5, dim=1)
                # Convert the indices to a numpy array
                predicted_indices = top_indices.cpu().numpy()
                predicted_labels = original_labels_mapping[predicted_indices]
                label_list=predicted_labels
        self.wg = label_list[0]
        print(self.wg)
        
    def get_question_context_faiss(self, batch, k, use_context=False):
        try:
            faiss_index, faiss_index_to_data_mapping, source_mapping = get_faiss_batch_index(batch)
            if use_context:
                result = find_nearest_neighbors_faiss(self.query, faiss_index, faiss_index_to_data_mapping, k, source_mapping= source_mapping, context=self.context)
            else:
                result = find_nearest_neighbors_faiss(self.query, faiss_index, faiss_index_to_data_mapping, k, source_mapping= source_mapping)
            
            if isinstance(result, list):
                    self.context = []
                    self.context_source = []
                    for i in range(len(result)):
                        index, data, source = result[i]
                        self.context.append(f"\nRetrieval {i+1}:\n...{data}...\nThis retrieval is performed from the document {source}.\n")
                        self.context_source.append(f"Index: {index}, Source: {source}")
            else:
                self.context = result
        except Exception as e:
            print(f"An error occurred while getting question context: {e}")
            print(traceback.format_exc())
            self.context = "Error in processing"

    
    