# final_project

In this phase, we experiment with the dataset provided with respect to three entities - location, user and images where each user has captured a set of images at certain
locations and has tagged them. It also provides two categories of information - some of the features are extracted for text (image tags, title) like TF, DF, TF-IDF scores
forming a set of textual descriptors and various models for images like CN, CM, HOG, etc which are extracted from the images and form a set of visual descriptors. These
descriptors help in creating a feature vector for each entity in the dataset and enable comparison between them by computing a similarity score using metrics like
Euclidean distance.
We primarily work on visual descriptors and graph models. Clustering/partitioning algorithms are explored. Top k ranking algorithms like pagerank and personalized pagerank
are also implemented. Search in multidimensional data space is supported well by an index structure, an example of which is Locality sensitive hashing (LSH). It is implemented
and query results are evaluated for an input consisting of a combination of visual descriptors given an image. We also evaluate KNN and Personalized pagerank based classification
algorithms for the set of images and labels provided.

