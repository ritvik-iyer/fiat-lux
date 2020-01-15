# Fiat-Lux
The Latin phrase "fiat lux" translates to "let there be light." In our increasingly digitized world, we must be wary about the news sources from which we consume information. As such, we aim to bring light to the underlying political advocacy and messaging within the news articles we read everyday. We aim to create a text classifier that is able to predict the media outlet which wrote a particular news articles.  

## Overview
We have built and tested several text classifiers trained on data from [FakeNewsCorpus](https://github.com/several27/FakeNewsCorpus). We filtered the dataset to only include political articles from Breitbart and the New York Times. For specifics into our methodology and conclusions, our research paper is available [here](https://drive.google.com/file/d/1LHyxv8Y6i90n6bfD-k5HI-Mm2p73wabx/view?usp=sharing). 

## Implementation

The classifier and other relevant scripts are implemented in python 3.6.

## Dependencies
-pandas
-nltk (Natural Language Toolkit)
-regex
-numpy 
-sklearn 
-pickle 

## Usage
To run our text classifier on an article of your choice:

 >>> python interactive.py 
    
You will be prompted to enter the name of a .txt file which contains the text of the article you want to classify. Please enter and save the text of the article you want to classify into a .txt file within the same folder as the interactive.py script. Then, when prompted, enter:
  >>> (filename).txt
  
