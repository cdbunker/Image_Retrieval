# Image_Retrieval

Here I use a Convolutional Neural Network to reduce MNIST images down to a 1x128 feature vector. I use triplet loss to make sure that, for each number class, this feature vector has a low intra-class difference while having a large inter-class difference. The final goal here is to do image retrieval.

Here is an example of image retrieval before running the images through the algorithm. The first image in the row is the query image while the rest are the nearest 10 neighbors in the original feature space.
![alt text](https://github.com/cdbunker/Image_Retrieval/blob/master/retrieval_no_encoding.png)

After encoding the images with the algorithm, the nearest neighbors are calculated again. Notice how much better the '2' class performs.
![alt text](https://github.com/cdbunker/Image_Retrieval/blob/master/retrieval_with_encoding.png)

Here is the t-SNE embedding of the original feature space.
![alt text](https://github.com/cdbunker/Image_Retrieval/blob/master/tsne_no_encoding.png)

Here is the t-SNE embedding of the encoding feature space. Notice that the classes have more space between them.
![alt text](https://github.com/cdbunker/Image_Retrieval/blob/master/tsne_with_encoding.png)
