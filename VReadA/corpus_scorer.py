from VReadA.scorer import get_scores_from_texts
from collections import OrderedDict
from glob import glob
import json
import math
import re
import os

# Updates the dictionary having n, s (temp variables),
# m (mean) and STDEV using the the new value.
def update(dictionary,val,indicator):
    # If none, initialize it with an empty dictionary.
    if dictionary is None:
        dictionary = {}
        dictionary['N'] = 0
        dictionary['S'] = 0
        dictionary['M'] = 0
        dictionary['STDEV'] = 0
        dictionary['REV'] = indicator

    n = dictionary['N']
    m = dictionary['M']
    s = dictionary['S']

    # Compute the new values
    n = n + 1
    m_prev = m
    m = m + (val - m) / n
    s = s + (val - m) * (val - m_prev)

    # Set the new values
    dictionary['N'] = n
    dictionary['S'] = s
    dictionary['M'] = m
    dictionary['STDEV'] = math.sqrt(s/n)
    return dictionary

# Get the sentence level scores by processing all the sentences present in
# the corpus directory using the nlp object and dump them in sentence_level_stats.txt
def accumulate_sentence_level_scores(directory,nlp,model_data):
    # Check if corpus directory is present
    # if not, then return status False and a message
    # that the directory is not valid
    if not os.path.isdir(directory):
        return False, directory + " is not a valid directory."
    
    # List of all files in the corpus
    file_list = glob(os.path.join(directory,"*"))

    # List of all the textss
    file_texts = []
    for file_name in file_list:
        filetext = open(file_name,"r").read()
        # Reduce multispaces to a single space
        filetext = re.sub(' +', ' ',filetext)
        # Remove new line characters
        filetext = re.sub('\n',' ',filetext)
        file_texts.append(filetext)

    # Call the get_scores_from_texts method to get the scores for 
    # all sentences in the texts
    sent_scores = get_scores_from_texts(file_texts,nlp,model_data)

    per_sentence_scores = {}
    for score_dictionary in sent_scores:
        # For each type of score in the dictionary, update it in per_sentence_score using
        # the update method
        for key in score_dictionary.keys():
            per_sentence_scores[key] = update(per_sentence_scores.get(key,None),score_dictionary[key][0],score_dictionary[key][1])
    
    store_dictionary  = {}
    store_dictionary["Per Sentence Scores"] = per_sentence_scores

    # Store the dictionary in a file
    file_name = directory + "/sentence_level_stats.txt"
    output_file = open(file_name,"w")
    output_file.write(json.dumps(store_dictionary))
    output_file.close()

    return True,"OK"

# Prints the sentence level stats by reading them from the file
def get_sentence_level_scores(folder_name):
    sentence_level_data = json.load(open("{}/sentence_level_stats.txt".format(folder_name)))
    print(sentence_level_data)
