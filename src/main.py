from Semantification import Preprocessing
from Clustering import density_Clustering
from Clustering import ClusterPurity

def main():

    dataloader= Preprocessing(PATH_TRANS_E= 'data/pre-trained/transE_fb15k_256dim.pkl', BASE_PATH_TRUTH = 'data/FB15k-237')

    semantifier=density_Clustering(dataloader.X_all, dataloader.y, dataloader.labels)
    

if __name__ == "__main__":
    main()