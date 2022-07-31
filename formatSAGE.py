from sys import settrace
from collections import Counter
from sklearn.preprocessing import MinMaxScaler
import gif

from json.tool import main
from memory_profiler import profile

import pandas as pd
from torch.nn import ReLU
from neo4j import GraphDatabase

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
from matplotlib import pyplot as plt

import numpy as np
from statistics import mean
import plotly.express as px
from sklearn import preprocessing
import coloredlogs, logging
import torch
import torch_geometric
import torch_geometric.transforms as T
from torch_geometric.data import HeteroData
from torch_geometric.loader import HGTLoader, NeighborLoader
from torch_geometric.nn import Linear, SAGEConv, Sequential, to_hetero, MetaPath2Vec
from torch_geometric.nn import HeteroConv, GCNConv, SAGEConv, GATConv,GATv2Conv, Linear, SuperGATConv
from torch_geometric.datasets import OGB_MAG
from torch_geometric.nn import HGTConv, Linear, HeteroLinear
from torch_geometric.nn import GATConv, Linear, to_hetero
import torch.nn.functional as F
from torchvision.ops.focal_loss import sigmoid_focal_loss
from tqdm import tqdm
import time
import wandb
from IPython.display import Image
import os
import plotly.graph_objects as go
from sklearn.model_selection import StratifiedKFold
from transformers import AutoTokenizer, AutoModel
from random import randint
from sentence_transformers import SentenceTransformer

coder_model = AutoModel.from_pretrained('GanjinZero/UMLSBert_ENG')
coder_tokenizer = AutoTokenizer.from_pretrained('GanjinZero/UMLSBert_ENG')

from segmentation_models_pytorch.losses import FocalLoss

pd.options.mode.chained_assignment = None  # default='warn'

torch.cuda.empty_cache()

mylogs = logging.getLogger(__name__)

num_of_neg_samples= 20
num_of_pos_samples= 20

seed = 10
data = None
print(seed)

class Connection:
    
    def fetch_data(self,query, params={}):
        with self.driver.session() as session:
            result = session.run(query, params)
            #return result
            return pd.DataFrame([r.values() for r in result], columns=result.keys())

    def __init__(self):
        self.driver = GraphDatabase.driver("bolt://127.0.0.1:17687", auth=("neo4j", "123456"))

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads):
        super().__init__()
        torch.manual_seed(seed)
        self.conv1 = GATConv((-1,-1), 64)  # TODO
        self.in1 = torch.nn.BatchNorm1d(64)
        self.conv2 = GATConv((-1,-1), 2)  # TODO
        # self.in2 = torch.nn.InstanceNorm1d(-1)
        # self.conv3 = GATConv((-1,-1), 2)

    def forward(self, x, edge_index):
        # x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = self.in1(x)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        # x = self.in2(x)
        # x = F.relu(x)
        #x = F.elu(x)
        # x = F.dropout(x, p=0.3, training=self.training)
        # x = self.conv3(x, edge_index)
        return x






def getData():
    conn = Connection()
    diagnosis_query="""MATCH (n:D_ICD_Diagnoses) where (n.long_title contains 'sepsis') or (n.long_title contains 'septicemia')  RETURN collect(n.icd9_code) as icd_arr""" 
    df_icd9 = conn.fetch_data(diagnosis_query)
    icd9_arr= df_icd9['icd_arr'][0]
    
    #hadm_query= """MATCH (n:Admissions)-[r:DIAGNOSED]-(m:D_ICD_Diagnoses) where m.icd9_code in """+str(icd9_arr)+""" RETURN n.hadm_id as hadm"""

    #hadm_query ="""MATCH (n:Note_Events) where n.category='Discharge summary' and ((toLower(n.text) contains toLower('sepsis')) or ((toLower(n.text) contains toLower('septic shock')) or ((toLower(n.text) contains toLower('severe sepsis')))))  RETURN collect(distinct n.hadm_id) as cols"""
    hadm_query= """MATCH (n:Note_Events) where n.category='Discharge summary' and (toLower(n.text) contains 'sepsis' or toLower(n.text) contains 'septic') RETURN collect(distinct n.hadm_id) as cols"""
    df_hadm = conn.fetch_data(hadm_query)
    hadm_arr=  df_hadm['cols'][0] #df_hadm['hadm'].tolist()  #

    #hadm_arr = hadm_arr[0:100]

    #print(hadm_arr)


    adm_pat_query = """MATCH (n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as expire, n.hadm_id as hadm_id"""

    df_pat = conn.fetch_data(adm_pat_query)
    df_grp = df_pat.groupby(['expire'])


    # temp_lst =[]
    # for x in [0,1]:
    #     if x<=0:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:num_of_neg_samples])
    #     else:
    #         grp = df_grp.get_group(x)
    #         temp_lst.extend(grp['hadm_id'].values.tolist()[0:num_of_pos_samples])
    # for x in [0,1]:
    #     grp = df_grp.get_group(x)
    #     temp_lst.extend(grp['hadm_id'].values.tolist()[0:1882])
        
    #hadm_arr = temp_lst


    print(len(hadm_arr))
    pat_arr= df_pat['patients'].tolist()

    pat_dat_query = """ MATCH (n:Patients) where n.subject_id in """+str(hadm_arr)+""" RETURN n.gender as gender, n.dob as birth, n.subject_id as patient_id"""

    #df_pat_data = conn.fetch_data(pat_dat_query)

    df_diagnosis_query = """MATCH (n:Admissions)-[r:DIAGNOSED]->(m:D_ICD_Diagnoses) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.hadm_id as hadm_id, n.hospital_expire_flag as expire, m.long_title as title"""

    df_diagnosis = conn.fetch_data(df_diagnosis_query)

    adm_query= """MATCH (n:Admissions) where n.hadm_id in """+str(hadm_arr)+""" RETURN n.subject_id as patients, n.hospital_expire_flag as label, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, n.hadm_id as hadm_id"""

    df_admission = conn.fetch_data(adm_query)
    

    # weights : and m.itemid in [50983,51221,50971,51249,51006,51265,50902,51301,50882,51250,50931,50912,51222,51279,51277,50868,51248,50960,50970,51237,51274,50893,51275,50804,50820,50821,50813,50818,50802]

    #and m.itemid in [50813,50931,50912,50868,50983,51237,51006,50885,50960,50902,50971,50820,50825]
    # high variance : [50813,50868,50885]
    #low variance: [50960,50971,50983]
    lab_query= """MATCH (x:Patients)-[xr:ADMITTED]-(n:Admissions)-[r:HAS_LAB_EVENTS]->(m:D_Lab_Items) where n.hadm_id in"""+str(hadm_arr)+""" and x.subject_id in """+str(pat_arr)+""" and duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours<= 24 RETURN ID(n) as start, ID(m) as end, r.value as value, r.valueUOM as units, date(r.charttime) as lab_time, n.hospital_expire_flag as label, n.marital_status as marital, n.ethnicity as ethnicity, n.religion as religion, m.fluid as fluid, m.itemid as itemid, m.category as category, m.label as lab_label,m.label as lab_name, n.hadm_id as adm_id,n.hadm_id as hadm_id, x.gender as gender, date(x.dob) as birth, date(n.admittime) as admit, duration.inSeconds(datetime(n.admittime), datetime(r.charttime)).hours as diff""" 

    df_lab = conn.fetch_data(lab_query)
    
    

    drug_query= """MATCH (n:Admissions)-[r:PRESCRIBED]->(m:DRUGS) where n.hadm_id in """+str(hadm_arr)+""" RETURN  ID(n) as start, ID(m) as end, r.STARTDATE as drug_start_date, r.ENDDATE as drug_end_date, r.dosage_val as dosage_val, r.dosage_unit as dosage_unit, r.generic_name as generic_name, m.name as drug_name, n.hadm_id as hadm_id"""
    # duration.inSeconds(datetime(r.STARTDATE), datetime(r.ENDDATE)).hours as drug_duration,
    #print(drug_query)

    df_drug =  conn.fetch_data(drug_query)

    return df_lab, df_drug, df_admission, df_diagnosis

def sentence_emd(sent):
    inputs = coder_tokenizer(sent,padding=True, truncation=True, max_length = 200, return_tensors='pt')
    sent_embed = np.mean(coder_model(**inputs).last_hidden_state[0].detach().numpy(), axis=0)
    return sent_embed


def map_edge_list(lst1):
    final_lst=[]
    set1= set(lst1)

    i=0
    lst1_new={}
    for val in set1:
        lst1_new[val]=i
        i=i+1
    
    return lst1_new

def create_train_val_test_mask(df):
    X = df.iloc[:,df.columns != 'label']
    Y = df['label']
   
  
 
    X_train_complete, X_test, y_train, y_test = train_test_split(X,df['label'].values.tolist(), test_size=0.1,random_state=seed,stratify=Y)
    X_train, X_val, y_trainval, y_testval = train_test_split(X_train_complete,y_train, test_size=0.1,random_state=seed,stratify=y_train)
    
    print("y_train: ",Counter(y_train).values()) 
    print("y_test: ",Counter(y_test).values()) 
    print("y_trainval: ",Counter(y_trainval).values()) 
    print("y_testval: ",Counter(y_testval).values()) 


    train_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    test_mask = torch.zeros(df.shape[0], dtype=torch.bool)
    val_mask = torch.zeros(df.shape[0], dtype=torch.bool)

    train_mask[X_train.index] = True
    test_mask[X_test.index] = True
    val_mask[X_val.index] = True


    conf_df = pd.DataFrame()
    conf_df['admmision_id']= X_test['admmision_id']
    conf_df['actual']= y_test



    return train_mask,val_mask,test_mask,conf_df

def train(model,optimizer,criterion,data):
      model.train()
      optimizer.zero_grad()  # Clear gradients.
      out = model(data.x_dict, data.edge_index_dict)  # Perform a single forward pass.
      mask = data['Admission'].train_mask
      loss = criterion(out['Admission'][mask], data['Admission'].y[mask])  # Compute the loss solely based on the training nodes.
      #print(out['Admission'][mask].shape)
      #print(data['Admission'].y[mask].shape)
      loss.backward()  # Derive gradients.
      optimizer.step()  # Update parameters based on gradients.
      return loss


def test(model,optimizer,criterion,mask,data):
      model.eval()
      out = model(data.x_dict, data.edge_index_dict)
      pred = out['Admission'].argmax(dim=1)  # Use the class with highest probability.
      correct = pred[mask] == data['Admission'].y[mask]  # Check against ground-truth labels.
      acc = int(correct.sum()) / int(mask.sum())  # Derive ratio of correct predictions.
      return acc,pred

def expandEmbeddings(df):
    df2 = pd.DataFrame(df['Embeddings'].tolist())
    return df2

#@profile
def main():
    model = None
    prev_mask = None
    global dataset

    heatmaps=[]
 
    #transform = T.ToUndirected(merge=True)
    if os.path.exists('MIMICDataObj.pt'):
        data =torch.load("MIMICDataObj.pt")
    else:

        st_time_nodes = time.time()

        df_labs, df_drugs, df_admission, df_diagnosis = getData()

        # enable to average the edges
        #df_labs = grp_labs(df_labs)

        print(df_drugs)

        end_time = time.time()

        print("Time for fetching data: ",end_time-st_time_nodes)

        df_labs = df_labs[df_labs['value'].notna()]

        df_labs['age'] = df_labs.apply(lambda e: (e['admit'] - e['birth']).days/365, axis=1)

        
        df_labs = df_labs[df_labs.age < 100 ]
        print(df_labs.shape)
        df_labs = df_labs[df_labs.age > 18 ]
        print(df_labs.shape)


        dict_start = map_edge_list(df_admission['hadm_id'].values.tolist())
    
        #vals = df_labs['adm_id'].values.tolist()

        df_admission['admmision_id'] = df_admission['hadm_id']

        
        label_encoder = preprocessing.LabelEncoder() 

        df_admission['ethnicity']= label_encoder.fit_transform(df_admission['ethnicity']) 
        df_admission['religion']= label_encoder.fit_transform(df_admission['religion']) 
        df_admission['marital']= label_encoder.fit_transform(df_admission['marital']) 

        df_labs['ethnicity']= label_encoder.fit_transform(df_labs['ethnicity']) 
        df_labs['lab_label'] = label_encoder.fit_transform(df_labs['lab_label'])
        df_labs['category'] = label_encoder.fit_transform(df_labs['category'])
        df_labs['marital']= label_encoder.fit_transform(df_labs['marital'])
        df_labs['gender']= label_encoder.fit_transform(df_labs['gender'])

        df_drugs['drug_name']  = label_encoder.fit_transform(df_drugs['drug_name'])

        

        df_diagnosis['Embeddings'] = df_diagnosis['title'].apply(lambda x:  np.array(sentence_emd(x)))
        df_diagnosis_features = expandEmbeddings(df_diagnosis)
        #df_diagnosis['Embeddings'] = torch.from_numpy(np.array(df_diagnosis['Embeddings']))
        #print(df_diagnosis)
        #df_diagnosis.to_csv('diagnosis.csv')



        df_labs['weight']= np.where(df_labs['label']<1, 1, 4)
        df_labs["adm_id"]= df_labs["adm_id"].map(dict_start)
        df_drugs["hadm_id"]= df_drugs["hadm_id"].map(dict_start)
        df_diagnosis["hadm_id"]= df_diagnosis["hadm_id"].map(dict_start)
        df_admission["hadm_id"] = df_admission["hadm_id"].map(dict_start)

        df_admission.index = df_admission['hadm_id']
        df_admission.sort_index()

    
        df_labs= df_labs[pd.to_numeric(df_labs['value'], errors='coerce').notnull()]
        df_labs['value'] = df_labs['value'].astype(float)
        df_labs = df_labs.reset_index(drop=True)

        df_drugs= df_drugs[pd.to_numeric(df_drugs['dosage_val'], errors='coerce').notnull()]
        df_drugs['dosage_val'] = df_drugs['dosage_val'].astype(float)
        df_drugs = df_drugs.reset_index(drop=True)

        df_labs['index_col'] = df_labs.index
        df_admission['index_col'] = df_admission.index
        df_drugs['index_col'] = df_drugs.index
        df_diagnosis['index_col'] = df_diagnosis.index

        mask_list = create_train_val_test_mask(df_admission)
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    for aggr in ['sum']:
        if not os.path.exists('MIMICDataObj.pt'):
            data = HeteroData()
            
            
            data['Admission'].x = torch.tensor(df_admission[['ethnicity','marital','religion']].values, dtype = torch.float).to(device)
            data['Admission'].y =  torch.tensor(df_admission['label'].values, dtype = torch.long).to(device)
            data['Admission'].train_mask = mask_list[0]
            data['Admission'].val_mask = mask_list[1]
            data['Admission'].test_mask = mask_list[2]
            data['Labs'].x = torch.tensor(df_labs[['category','lab_label','age','gender']].values, dtype = torch.float).to(device)
            data['Admission', 'has_labs', 'Labs'].edge_index = torch.tensor(df_labs[['adm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Admission', 'has_labs', 'Labs'].edge_attr  = torch.tensor(df_labs[['value','diff']].values, dtype=torch.long).t().contiguous().to(device)
            print(df_drugs[['dosage_val']].values)
            data['Drugs'].x = torch.tensor(df_drugs[['drug_name']].values, dtype = torch.float).to(device)
            data['Admission', 'has_drugs', 'Drugs'].edge_index = torch.tensor(df_drugs[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            data['Admission', 'has_drugs', 'Drugs'].edge_attr  = torch.tensor(df_drugs[['dosage_val']].values, dtype=torch.long).t().contiguous().to(device)
            #df_diagnosis.iloc[:,4:].drop('index_col',axis=1).values
            data['Diagnosis'].x = torch.tensor(df_diagnosis_features.values,dtype = torch.float).to(device)
            data['Admission', 'has_diagnosis', 'Diagnosis'].edge_index = torch.tensor(df_diagnosis[['hadm_id','index_col']].values.tolist(), dtype=torch.long).t().contiguous().to(device)
            

            data.num_node_features = 3
            data.num_classes = len(df_labs['label'].unique())
            data = T.ToUndirected()(data.to(device))
            data = T.NormalizeFeatures()(data.to(device))
            #data = T.RandomNodeSplit()(data)
            dataset = data.to(device)

            data = dataset.to(device)
            print(data)
            if not os.path.exists('MIMICDataObj.pt'):
                torch.save(data,'MIMICDataObj.pt')
            # train_loader = NeighborLoader(
            #     data,
            #     # Sample 15 neighbors for each node and each edge type for 2 iterations:
            #     num_neighbors=[4] * 2,
            #     # Use a batch size of 128 for sampling training nodes of type "paper":
            #     batch_size=8,
            #     input_nodes=('Admission', data['Admission'].train_mask),
            # )
            # batch = next(iter(train_loader))  
        
        if data:
            model = GAT(hidden_channels=8, heads=8)
            print(model)
            model = to_hetero(model, data.metadata(), aggr=aggr).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=5e-5, weight_decay=5e-4)
            criterion = torch.nn.CrossEntropyLoss(weight=torch.tensor([0.20, 0.80])).to(device)  #
            # criterion =  FocalLoss(mode="binary", alpha=0.25, gamma=2)
            for epoch in range(1, 1000):
                loss = train(model,optimizer,criterion,data)
                train_acc,pred_train = test(model,optimizer,criterion,data['Admission'].train_mask,data)
                val_acc,pred_val = test(model,optimizer,criterion,data['Admission'].val_mask,data)
                test_acc,pred_test = test(model,optimizer,criterion,data['Admission'].test_mask,data)
                print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}, Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}')
            
            mask_train = data['Admission'].train_mask
            cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_train], pred_train[mask_train].cpu())
            print("train cfm: ",cf_matrix)


            mask_test = data['Admission'].test_mask
            cf_matrix = confusion_matrix(data['Admission'].cpu().y[mask_test], pred_test[mask_test].cpu())
            print("test cfm: ",cf_matrix)
            
    
        
def seed_everything(seed=seed):                                                  
       #random.seed(seed)                                                            
       torch.manual_seed(seed)                                                      
       torch.cuda.manual_seed_all(seed)                                             
       np.random.seed(seed)                                                         
       os.environ['PYTHONHASHSEED'] = str(seed)                                     
       torch.backends.cudnn.deterministic = True                                    
       torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    seed_everything()
    main()