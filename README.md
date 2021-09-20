## Towards Unsupervised Semantification with Knowledge Graph Embedding

This reposity contains the source code of paper *Towards Unsupervised Semantification with Knowledge Graph Embedding*

### Summary:
* We propose a fully unsupervised approach for predicting entities types in knowledge graphs. Our approach contains four steps as shown in Figure 1:
  * Data preprocessing: Given an input tabular data (csv), we first preprocess the data and tranform it to a knowledge graph (RDF triples). Each triple represent an information about entity `<subject, predicate, object>`
   * In this step, we employ a knowledge graph embedding to represent entitiens and their relations in the same semantic space. 
   * Then, we employ a density-based clustering approach (hdscan) to detect clusters of entites. Each cluster should have entities with similar properities.
   * Finally, we sample few entities (nearby cluster centroid) and ask human annotators to label them manually. Afterwards, we assign the major type to all entities in the same cluster. 

<p align="center">
<img src="src/Figures/pipeline2.png" alt="">
</p>
<p align="center">Fig. 1 Pipleine of our Semantification Process</p>

***
### Installation:

* Python: we develop our approach in Python3.6, check https://www.python.org/downloads/release/python-360/ for detailed installation.

* Dependencies: we use different libraries to load, develop and visualize our results, please install them from requirements.txt file via `pip install -r requirements.txt`

* hdbscan: We use the HDBSCAN clustering library (https://hdbscan.readthedocs.io/en/latest/how_hdbscan_works.html) to compute density-based clusters. Please check https://pypi.org/project/hdbscan/ to install hdbscan in Python3.6
* Vectograph: This library is used to construct a knowledge graph (KG) for an input tabular data (CSV). For more details, please follow the instructions described in https://github.com/dice-group/Vectograph 

* Graphvite: This library used to generate the embeddings represenation for an input KG. More information can be found here https://graphvite.io/docs/latest/install.html

***
### Dataset:
* FB15k-237: We experiment our approach on the benchmark dataset (FB15k-237) for type and link prediction tasks. This dataset is a subset of Freebase Knowledge Graph contains 310,116 triples with 14,951 entities and 237 relations. The source can be found in `data` folder.

***
### How to run:
There are two folders for our approach implementation: 
* `src`: contains the source-code in Python 3.6
* `notebooks`: contains three notebooks for our experiment on FB15k-237. Each notebook evaluates our approach with different KG embeddings (transE, distMult, and rotatE).

### Results:
Fig. 2 shows an example of {transE embeddings, hdbscan clustering} results using t-SNE projection. We plotted entities in six types (education, film, location, music, people, and soccer). It's clearly seen that, entities with same type (e.g. film --in orange color--), cluster well based on their embeddings representation.

<p align="center">
<img src="src/Figures/fb15k-transE-full.png" alt="" width="400" height="300">
 </p>
<p align="center"> Fig. 2 t-SNE visualization of semantification process on FB15k-237 with TransE embedding.</p>

For more visualization results with different embeddings, please check `src/Figures`
***
### Aknowledgment: 
This work was supported by the German Federal Ministry of Education and Research within the DAIKIRI project (grant no: 01IS19085).
***
### Citation
TBD


If you have any further questions/suggestions, please contact hamada.zahera@upb.de
