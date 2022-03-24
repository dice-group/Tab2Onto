from Semantification import Preprocessing
from Clustering import density_Clustering

def main():

    dataloader= Preprocessing(PATH_TRANS_E= 'data/pre-trained/transE_fb15k_256dim.pkl', BASE_PATH_TRUTH = 'data/FB15k-237')

    # For example, apply density-based clustering (hdbscan) with FB15k-237 with TransE embedding
    density_Clustering(dataloader.X_all, dataloader.y, dataloader.labels)


if __name__ == "__main__":
    main()