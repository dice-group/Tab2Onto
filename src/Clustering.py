#In this step, we group entities with similar properites (i.e., based on their embedding representations) into clusters. 
#Each group should have similar entities --> similar types.

#We employ a density-based clustering (hdbscan) to detect entities cluster based on their density in the embedding space.
#We use the implementation of hdbscan clustering library. For more information/install,
#please check https://hdbscan.readthedocs.io/en/latest/index.html


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from clustering_evaluation import ClusterPurity

#HDBSCAN requires two main hyper-parameters:
# 1) epslion, which specify the area within it, there should be a min_samples to consider a point a core point. 
# We use the eplow approach to find a best value for epslion.

from sklearn.neighbors import NearestNeighbors


class ClusterPurity:

    def __init__(self):
        """
        param:
        return
        """

    def purity_score(self, y_true, y_pred):
        """
        param: y_true: the ground_truth labels of clusters. 
               y_pred: the predicted cluster labels.
        return: the purity score of clustering
        """
        # compute contingency matrix (also called confusion matrix)
        contingency_matrix = metrics.cluster.contingency_matrix(y_true, y_pred)
        # return purity
        return np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)


class density_Clustering: 

    def __init__(self, X_all, y_all, labels):

        self.X_all=X_all
        self.y_all=y_all
        self.labels=labels

        self.plot_epslion() # plot eblow to find the best value of epslion in HDBSCAN

        self.distance_matrix=compute_pairwiseDist()
    
        self.hdbscan_clusterer=hdbscan.HDBSCAN(algorithm='best', alpha=0.1, metric='precomputed', cluster_selection_method='leaf',
                                        min_samples=10, min_cluster_size=700, core_dist_n_jobs=-1,allow_single_cluster=True,
                                        cluster_selection_epsilon=0.9)

        hdbscan_clusterer.fit(distance_matrix)

        self.cluster_labels= hdbscan_clusterer.labels_
        self.cluster_probabilities=hdbscan_clusterer.probabilities_


        #visualize labeled entities using t-SNE projection
        visualize_entities()

        #Entity Typing Evaluation: accuracy, precision, recall, F1
        evaluation()


# compute the distance between entities using cosine
def compute_pairwiseDist(self):
    X_all_double=X_all.astype(np.double)
    distance_matrix = pairwise_distances(X_all_double, metric='cosine')

    return distance_matrix


def plot_epslion(self):
    # final optimal value for cluster epsilon
    neigh = NearestNeighbors(n_neighbors=5)
    nbrs = neigh.fit(X_all)
    distances, indices = nbrs.kneighbors(X_all)

    distances = np.sort(distances, axis=0)
    distances = distances[:,-1]
    plt.plot(distances)


#Sampling Entities for Labeling:
#In the following, we present our strategy to select entities based on its membership in l clusters.
#We compute the cluster probabilies for all entities (cluster_probabilities). For each cluster, we select entities with high values >= 0.9 for labeling.
#We present the selected entities (with their RDF triples) to human expers for labeling.
#Finally, we propagate the most frequent type in each cluster to all entities.

def visualize_entities(self): 
    # propagate the most frequent type in the cluster to all entities. 
    df_tmp = pd.DataFrame({'pred_hdbscan': self.labels, 'y_all': self.y_all})
    pred_hdbscan = df_tmp.groupby('pred_hdbscan').transform(lambda x: x.mode().iloc[0]).to_numpy().reshape(-1)

    plt.figure(figsize=(6, 5))
    X_2d = TSNE(random_state=42).fit_transform(self.X_all)

    label_ids = range(len(self.labels))
    colors=['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown']

    for i, c, label in zip(label_ids, colors, self.labels):    
        plt.scatter(X_2d[pred_hdbscan == i, 0], X_2d[pred_hdbscan == i, 1], c=c, label=label, s=1)

    plt.legend()    
    plt.savefig('/src/Figures/fb15k-transE-hdbscan.png', dpi=600, bbox_inches='tight',pad_inches=0)    
    plt.show()


def evaluation(self):
    accuracy = accuracy_score(self.y_all, self.cluster_labels)
    print('Accuracy: %f' % accuracy)

    precision = precision_score(self.y_all, self.cluster_labels, zero_division=0, average='weighted')
    print('Precision: %f' % precision)

    recall = recall_score(self.y_all, self.cluster_labels, average='weighted')
    print('Recall: %f' % recall)

    f1 = f1_score(self.y_all, self.cluster_labels, average='weighted')
    print('F1 score: %f' % f1)

    evaluator=ClusterPurity() # to compute cluster purity
    print ('Purity: ' , evaluator.purity_score(y_true=self.y_all, y_pred=self.cluster_labels))


