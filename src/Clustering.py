import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from clustering_evaluation import ClusterPurity

import hdbscan
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering

from sklearn.metrics.pairwise import pairwise_distances


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

        self.distance_matrix=self.compute_pairwiseDist()
    
        self.hdbscan_clusterer=hdbscan.HDBSCAN(algorithm='best', alpha=0.1, metric='precomputed', cluster_selection_method='leaf',
                                        min_samples=10, min_cluster_size=700, core_dist_n_jobs=-1,allow_single_cluster=True,
                                        cluster_selection_epsilon=0.9)

        self.hdbscan_clusterer.fit(self.distance_matrix)

        self.cluster_labels= self.hdbscan_clusterer.labels_
        self.cluster_probabilities=self.hdbscan_clusterer.probabilities_
   
        #Entity Typing Evaluation: accuracy, precision, recall, F1
        self.evaluation()

    # compute the distance between entities using cosine
    def compute_pairwiseDist(self):
        X_all_double=self.X_all.astype(np.double)
        distance_matrix = pairwise_distances(X_all_double, metric='cosine')

        return distance_matrix

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


class Centroid_Clustering: 
    def __init__(self, X_all, y_all, labels):

        self.X_all=X_all
        self.y_all=y_all
        self.labels=labels    
        self.kmeans = KMeans().fit(self.X_all)
        self.cluster_labels= self.kmeans.predict(self.y_all)

        self.evaluation()
        
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



class Agglomerative_Clustering:

    def __init__(self, X_all, y_all, labels):

        self.X_all=X_all
        self.y_all=y_all
        self.labels=labels    
        
        self.aggClustering = AgglomerativeClustering()
        self.cluster_labels= self.aggClustering.fit_predict(self.X_all)

        self.evaluation()

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

  