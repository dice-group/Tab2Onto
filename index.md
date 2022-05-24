This page provides the code and datasets for our ESWC poster: 
<p>
<h3> Tab2Onto: Unsupervised Semantification with Knowledge Graph Embeddings </h3> 
</p>

Cited as:
```
@INPROCEEDINGS
{zahera2022tab2onto, 
author = "Hamada M. Zahera, Stefan Heindorf, Stefan Balke, Jonas Haupt, 
Martin Voigt, Carolin Walter, Fabian Witter and Axel-Cyrille Ngonga Ngomo", 
title = "Tab2Onto: Unsupervised Semantification with Knowledge Graph Embeddings",
booktitle = "The Semantic Web: ESWC 2022 Satellite Events", 
year = "2022", series = "Springer"}
``` 

### Abstract
>"A large amount of data is generated every day by various systems and applications. In many cases, this data comes into a tabular format, which lacks semantic representation and poses new challenges in data modeling. It is necessary to elevate the data into a richer representation, such as a knowledge graph adhering to an ontology. This will assist in the development of data modeling and integration. We propose an unsupervised approach, Tab2Onto, for learning ontologies from tabular data using knowledge graph embeddings, clustering, and a human in the loop. We conduct a set of experiments to investigate our approach on a benchmarking dataset from a medical domain to learn ontology of diseases."

### How it works:
>Our approach includes five steps:
 >- (a) Data preprocessing: Given input tabular data (CSV), we first preprocess the data and transform it into a knowledge graph (RDF triples). Each triple describes information about an entity in form `<subject, predicate, object>`
 >* (b) Knowledge Graph Embedding: in this step, we represent entitiens and their relations into one semantic space. 
 >- (c) Clustering: we use a density-based clustering approach to detect clusters of entites. Each cluster contains entities with similar properties.
 >- (d) Human-In-The-Loop: we incorporate a domain expert as a human-in-the-loop to label clusters based on the properities of its entities. 
 >- (e) Finally, we populate the assigned label (i.e, class) to all entities within the same cluster.

### Poster
[Display ESWC-2022 Poster](https://github.com/dice-group/Tab2Onto/blob/main/Tab2Onto-Poster.pdf)
### Code: 
The source code and datasets used in our paper via

`git clone https://github.com/dice-group/Tab2Onto.git`

***
### Aknowledgment: 

This work has been supported by the German Federal Ministry for Economic Affairs and Climate Action (BMWK) within the project RAKI under the grant no 01MD19012B and by the German Federal Ministry of Education and Research (BMBF) within the project DAIKIRI under the grant no 01IS19085B.

If you have any further questions/suggestions, please contact `hamada.zahera@upb.de`
