# Fuzzy Topic Modeling - methods derived from Fuzzy Latent Semantic Analysis
This is the Python code to train Fuzzy Latent Semantic Analysis-based topic models. The details of the original FLSA model can be found [here](https://link.springer.com/article/10.1007/s40815-017-0327-9). With my group, we have formulated two alternative topic modeling algorithms 'FLSA-W' and 'FLSA-V' , which are derived from FLSA. Once the paper is published (it has been accepted), we will place a link here too. 

## Table of contents
1. Introduction to Topic Modeling
2. Explanation algorithms
3. Getting started
  * FLSA & FLSA-W
  * FLSA-W
    * Instructions to get map_file from Vosviewer
4. Class methods
5. Dependencies

## Introduction to Topic Modeling
Topic modeling is a popular task within the domain of Natural Language Processing (NLP). Topic modeling is a type of statistical modeling for discovering the latent 'topics' occuring in a collection of documents. While humans typically describe the topic of something by a single word, topic modeling algorithms describe topics as a probability distribution over words. 

Various topic modeling algorithms exist, and one thing they have in common is that they all output two matrices:
1. Probability of a word given a topic. This is a *M x C (vocabulary size x number of topics)* matrix. 
2. Probability of a topic given a document. This is a *C x  N (number of topics x number of documents)* matrix.

From the first matrix, the top *n* words per topic are taken to represent that topic. 

On top of finding the latent topics in a text, topic models can also be used for more expainable text classification. In that case, documents can be represented as a *'topic embedding'*; a *c*-length vector in which each cell represents a topic and contains a number that indicates the extend of which a topic is represented in the document. These topic embeddings can then be fed to machine learning classification models. Some machine learning classification models can show the weights they assigned to the input variables, based on which they made their decisions. The idea is that if the topics are interpretable, then the weights assigned to the topics reveal why a model made its decisions. 

## Explanation algorithms
The general approach to the algorithm(s) can be explained as follows:

1. Create a local term matrix. This is a *N x M (number of documents x vocabulary size)* matrix that gives the count of each word *i* in document *j*.
2. Create a global term matrix in which the words from different documents are also related to each other (the four options for weighting in the class are: 'normal', 'entropy','idf','probidf'). 
3. Project the data in a lower dimensional space (we use singular value decomposition).
4. Use fuzzy clustering to get the partition matrix.
5. Use Bayes' Theorem and matrix multiplication to get the needed matrices. 

### FLSA
The original FLSA approach aims to find clusters in the projected space of documents. 

### FLSA-W
Documents might contain multiple topics, making them difficult to cluster. Therefore, it might makes more sense to cluster on words instead of documents. That is what what we do with FLSA-W(ords). 

### FLSA-V
FLSA-W clusters on a projected space of words and implicitly assumes that the projections ensure that related words are located nearby each other. However, there is no optimization algorithm that ensures this is the case. With FLSA-V(os), we use the output from [Vosviewer](https://www.vosviewer.com/) as input to our model. Vosviewer is an open-source software tool used for bibliographic mapping that optimizes its projections such that related words are located nearby each other. Using Vosviewer's output, FLSA-V's calculations start with step 4 (yet, step 1 is used for calculating some probabilities). 

## Getting started
Many parameters have default settings, so that the algorithms can be called only setting the following two variables:
* **input_file**, The data on which you want to train the topic model.
    * *Format*: list of lists of tokens.
    * *Example*: [['this','is','the','first','document'],['why','am','i','stuck','in','the','middle'],['save','the','best','for','last']].

* **num_topics**, The number of topics you want the topic model to find. 
    * *Format*: int (greater than zero).
    * *Example*: 15.

Suppose, your data (list of lists of strings) is called `data` and you want to run a topic model with 10 topics. Run the following code to get the two matrices: 

    flsa_model = FLSA(input_file = data, num_topics = 10)
    prob_word_given_topic, prob_topic_given_document = flsa_model.get_matrices()

To see the words and probabilities corresponding to each topic, run:

    flsa_model.show_topics()

Below is a description of the other parameters per algorithm.

### FLSA & FLSA-W
* *num_words*, The number of words (top-*n*) per topic used to represent that topic. 
    * *Format*: int (greater than zero).
    * *Default value*: 20

* *word_weighting*, The method used for global term weighting (as describes in step 2 of the algorithm)
    * *Format*: str (choose between: 'entropy', 'idf', 'normal', 'probidf').
    * *Default value*: 'normal'

* *cluster_method*, The (fuzzy) cluster method to be used.
    * *Format*: str (choose between: 'fcm', 'gk', 'fst-pso').
    * *Default value*: 'fcm'

* *svd_factors*, The number of dimensions to project the data into. 
    * *Format*: int (greater than zero).
    * *Default value*: 2.

### FLSA-V
* *map_file*, The output file from Vosviewer. 
    * *Format*: pd.DataFrame (The Dataframe needs to contain the following columns: '*id*','*x*','*y*')
    * *Example*:  

| id | x | y |
| --- | --- | --- |
| word_one | -0.4626 | 0.8213 |
| word_two | 0.6318 | -0.2331 |
| ... | ... | ... |
| word_M | 0.9826 | 0.184 |


* *num_words*, The number of words (top-*n*) per topic used to represent that topic. 
    * *Format*: int (greater than zero).
    * *Default value*: 20

* *cluster_method*, The (fuzzy) cluster method to be used.
    * *Format*: str (choose between: 'fcm', 'gk', 'fst-pso').
    * *Default value*: 'fcm'

 #### Instructions to get map_file from Vosviewer

 1. Create a tab-separated file from your dataset in which you show for each word how often it appears with each other word. <br>
    Format: *Word_1* \<TAB\> *Word_2* \<TAB\> Frequency.<br>
    (Since this quickly leads to an unproccesable number of combinations, we recommend using only the words that appear in at least *x* documents; we used 100). 
 2. [Download Vosviewer](https://www.vosviewer.com/download). 
 3. *Vosviewer* \> *Create* \> *Create a map based on text data* \> *Read data from VOSviewer files* <br>
    Under 'VOSviewer corpus file (required)' submit your .txt file from step 1 and click 'finish'. 
 4. The exported file is a tab-separated file, and can be loaded into Python as follows: <br>
    Suppose the file is called `map_file.txt`: <br>
    `map_file = pd.read_csv('<DIRECTORY>/map_file.txt', delimiter = "\t")`
 5. Please check the [Vosviewer manual](https://www.vosviewer.com/documentation/Manual_VOSviewer_1.6.8.pdf) for more information. 
 
 ## Class Methods
 
 ## Dependencies
numpy == 1.19.2 <br>
pandas == 1.3.3 <br>
sparsesvd == 0.2.2 <br>
scipy == 1.5.2 <br>
pyfume == 0.2.0 <br>
