import pickle
import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

class Preprocessing: 

# By the default, this semantification process is configured to work with TransE KG model 
# on FB15k-237 dataset. 
# You can can the KG model and the dataset using parameters PATH_TRANS_E and BASE_PATH_TRUTH

    def _init__(self, PATH_TRANS_E= 'data/pre-trained/transE_fb15k_256dim.pkl', BASE_PATH_TRUTH = 'data/FB15k-237'):
        # transe_fb15k-237.pkl: pre-trained model of fb15k.
        with open(PATH_TRANS_E, "rb") as fin:
        model = pickle.load(fin)

        self.entity2id = model.graph.entity2id
        self.__init__relation2id = model.graph.relation2id

        self.entity_embeddings = model.solver.entity_embeddings
        self.relation_embeddings = model.solver.relation_embeddings

        self.ground_truth=load_Types()

        #get (X,y) dataset
        self.X_all, self.y, self.labels=  filter_topTypes()

    #Get the ground-truth types (labels) from the dataset
    def load_Types(self): 

        #extract ground-truth types:
        fb_train=pd.read_csv(self.BASE_PATH_TRUTH + '/train.txt', sep='\t', header=None, index_col=0)
        fb_valid=pd.read_csv(self.BASE_PATH_TRUTH + '/valid.txt', sep='\t', header=None, index_col=0)
        fb_test=pd.read_csv(slef.BASE_PATH_TRUTH + '/test.txt', sep='\t', header=None, index_col=0)

        fb_df=pd.concat([fb_train, fb_valid, fb_test])
        fb_df['type']= fb_df[1].apply(lambda x: x.split('/')[1])

        #combine entities with their types:

        ground_truth={}
        for entity_id in self.entity2id.keys():
            if entity_id in fb_df.index:
                if isinstance(fb_df.loc[entity_id, 'type'], pd.core.series.Series): 
                    ground_truth[entity_id]=fb_df.loc[entity_id, 'type'][0]
                else:
                    ground_truth[entity_id]=fb_df.loc[entity_id, 'type']
            else:
                ground_truth[entity_id]='unknown' # for missed types

        return ground_truth

    # Filter top 6 types from the FB15k-237 dataset
    # top_types=['people', 'film', 'location', 'music', 'soccer', 'education']

    def filter_topTypes(self): 
        entity_embedding_filter=[]
        y_true_filter=[]

        top_types=['people', 'film', 'location', 'music', 'soccer', 'education']

        for k, value in self.ground_truth.items():
            if value in top_types:        
                entity_embedding_filter.append(self.entity_embeddings[self.entity2id[k]])
                y_true_filter.append(value)
        
        X_all = np.asarray(entity_embedding_filter)

        #encode y_labels as one-hot:
        encoder = LabelEncoder()
        y_all = encoder.fit_transform(y_true_filter)
        labels = encoder.classes_.tolist()

        return X_all, y_all, labels
    










