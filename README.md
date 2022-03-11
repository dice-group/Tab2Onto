## Tab2Onto: Unsupervised Semantification with Knowledge Graph Embedding

### Overview:
We propose, Tab2Onto, an unsupervised approach for learning ontologies from tabular data. Our approach includes five steps as shown in Figure 1:

  (a) Data preprocessing: Given input tabular data (CSV), we first preprocess the data and transform it into a knowledge graph (RDF triples). Each triple describes information about an entity in form `<subject, predicate, object>`

   (b) Knowledge Graph Embedding: in this step, we represent entitiens and their relations into one semantic space. 

   (c) Clustering: we use a density-based clustering approach to detect clusters of entites. Each cluster contains entities with similar properties.
   (d) Human-In-The-Loop: we incorporate a domain expert as a human-in-the-loop to label clusters based on the properities of its entities. 

   (e) Finally, we populate the assigned label (i.e, class) to all entities within the same cluster.

<p align="center">
<img src="src/Figures/pipeline.jpeg" alt="">
<p align="center">
Tab2Onto pipleine for our semantification process</p>

---
### Installation:
please install them from requirements.txt file via `pip install -r requirements.txt`

***

### Dataset:
* FB15k-237:
this dataset is a subset of Freebase Knowledge Graph contains $310,116$ triples with $14,951$ entities and $237$ relations. The source can be found in `data` folder. we evaluated our approach on FB15K dataset to assess the performance for predicting types of entities (e.g. movie, person , organization) using embedding-based clustering. 
As  an example of {transE embeddings, hdbscan clustering} results using t-SNE projection. We plotted entities in six types (education, film, location, music, people, and soccer). It's clearly seen that, entities with same type (e.g. film --in orange color--), cluster well based on their embeddings representation.

<p align="center">
<img src="src/Figures/fb15k-transE-full.png" alt="" width="400" height="300">
<p align="center"> t-SNE visualization of semantification process on FB15k-237 with TransE embedding.</p>

  For more visualization results with different embeddings, please check `src/Figures`

* Lymphography Data: 
we investigated our full pipeline on the SML-Bench dataset, Lymphography. We processed all five steps to convert orginal lymphography data from tabular format to an ontology. The learned ontology are saved as an OWL in RDF/XML format.

***
### How to run:
There are two folders for our approach implementation: 
* `src`: contains the source-code in Python 3.6
* `notebooks`: contains three notebooks for our experiment on FB15k-237 and Lymphography datasets.

***
### Aknowledgment: 
This work was supported by the German Federal Ministry of Education and Research within the DAIKIRI project (grant no: 01IS19085).
***
### Citation
TBD


If you have any further questions/suggestions, please contact hamada.zahera@upb.de
