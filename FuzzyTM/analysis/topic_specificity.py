# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 16:05:32 2021

@author: 20200016
"""

from collections import Counter
import statistics

class TopicSpecificity:
    """
    A class to calculate specificity measures of words within given topics.
    
    Attributes:
        topics (list): A list of topics, each represented by a list of words or tuples.
        data (list): A list of documents with word frequencies.
        word_dict (Counter): A counter mapping words to their frequencies in `data`.
        sorted_word_count (list): A list of tuples with word frequencies sorted in descending order.
        sorted_counts (list): A list of word counts sorted in descending order.
        num_words (int): The number of words in a topic.
    """
    def __init__(
            self,
            topics,
            data,
            ):
        assert isinstance(topics, list), "'topics' is not a list."
        assert isinstance(data, list), "'data' is not a list."
        if isinstance(topics[0], tuple):
            self.topics = self.__transform_lda_output_to_topic_list__(topics)
        else:
            self.topics = topics
        self.data = data
        self.word_dict, self.sorted_word_count = self._sorted_word_count__(data)
        self.sorted_counts = [tup[1] for tup in self.sorted_word_count]
        self.num_words = len(self.topics[0])


    def specificity_word(
            self, word, relative = True
            ):
        """
        Calculate the specificity of a word either relative to the total word count or as an absolute rank.
        
        Parameters:
            word (str): The word to calculate specificity for.
            relative (bool): If True, calculate specificity relative to the total word count.
        
        Returns:
            float: The relative specificity if `relative` is True; otherwise, the absolute rank.
        """
        
        if relative:
            return (self.sorted_counts.index(self.word_dict[word])+1)/len(self.sorted_counts)
        else:
            return (self.sorted_counts.index(self.word_dict[word])+1)


    def word_rank(
            self, word
            ):
        """
        Get the rank of a word based on its frequency.
        
        Parameters:
            word (str): The word to get the rank for.
        
        Returns:
            int: The rank of the word.
        """
        
        return next(i for i, (w, *_) in enumerate(self.sorted_word_count) if w == word)


    def word_frequency(
            self, word
            ):
        """
        Get the frequency of a word in the data.
        
        Parameters:
            word (str): The word to get the frequency for.
        
        Returns:
            int: The frequency of the word.
        """
        
        return next(i for (w,i) in self.sorted_word_count if w == word)


    def average_rank(self):
        """
        Calculate the average rank of all words.
        
        Returns:
            float: The average rank of the words.
        """
        count = 0   
        for tup in self.sorted_word_count:
            count += tup[1]
       
        total_count = 0
        idx = 0
        
        while total_count < count/2:
            total_count+=self.sorted_word_count[idx][1]
            idx+=1
        return idx


    def average_frequency(self):
        """
        Calculate the average frequency of all words.
        
        Returns:
            float: The average frequency of the words.
        """
        return self.sorted_word_count[self.average_rank()][1]
  

    def topic_specificities_one_topic(
            self, topic_index, relative = True
            ):
        """
       Calculate the specificities of words in one topic.
       
       Parameters:
           topic_index (int): The index of the topic to calculate specificities for.
           relative (bool): If True, calculate specificities relative to the total word count.
       
       Returns:
           list: A list of specificities for the words in the topic.
       """
        
        word_specificities = []
        
        for word in self.topics[topic_index]:
            word_specificities.append(self.specificity_word(word, relative))
        
        return word_specificities


    def mean_topic_specificity_one_topic(
            self, topic_index, relative = True
            ):
        """
        Calculate the mean specificity of words in one topic.
        
        Parameters:
            topic_index (int): The index of the topic to calculate mean specificity for.
            relative (bool): If True, calculate mean specificity relative to the total word count.
        
        Returns:
            float: The mean specificity of the words in the topic.
        """
        specificities = self.topic_specificities_one_topic(topic_index, relative = relative)
        return sum(specificities)/len(specificities)


    def standard_deviation_topic_specificity_one_topic(
            self, topic_index, relative = True
            ):
        """
        Calculate the standard deviation of specificities for words in one topic.
        
        Parameters:
            idx (int): The index of the topic.
            relative (bool): If True, calculate specificities relative to the total word count.
        
        Returns:
            float: The standard deviation of specificities for the topic.
        """
        specificities = self.topic_specificities_one_topic(topic_index, relative = relative)
        return statistics.stdev(specificities)


    def topic_specificities_all_topics(
            self, relative = True
            ):
        """
        Calculate the specificities of all topics.
        
        Parameters:
            relative (bool): If True, calculate specificities relative to the total word count.
        
        Returns:
            list: A list containing specificities for all topics.
        """
        topic_list = []
        for i in range(len(self.topics)):
            topic_list.append(self.topic_specificities_one_topic(i, relative = True))
        return topic_list


    def mean_topic_specificity_all_topics(
            self, relative = True
            ):
        """
        Calculate the mean specificity across all topics.
        
        Parameters:
            relative (bool): If True, calculate mean specificity relative to the total word count.
        
        Returns:
            float: The mean specificity across all topics.
        """
        topic_specificities = self.topic_specificities_all_topics(relative = relative)
        count = 0
        total_words = 0
        
        for topic in topic_specificities:
            count += sum(topic)
            total_words += len(topic)
        return count/total_words
    
    
    def __transform_lda_output_to_topic_list__(
            self, topics
            ):
        """
        Private method to transform LDA output to a standard topic list.
        
        Parameters:
            topics (list): LDA output to be transformed.
        
        Returns:
            list: A list of topics with each topic represented as a list of words.
        """
        all_topics = []
        for i, top in enumerate(topics):
            topic_list = []
            for word in top[1].split('+'):
                topic_list.append(word.split('"')[1])
            all_topics.append(topic_list)
        return all_topics


    def _sorted_word_count__(
            self, data
            ):
        """
        Private method to get a sorted word count from the provided data.
        
        Parameters:
            data (list): The data to count words from.
        
        Returns:
            tuple: A counter of words and a list of word-count tuples sorted by frequency.
        """
        word_dict = Counter()
        for doc in data:
            word_dict.update(doc)
        return word_dict, word_dict.most_common()