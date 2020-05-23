from textstat.textstat import textstatistics, easy_word_set
from spacy import load
from VReadA.perplexity import get_perplexity
from collections import OrderedDict
import json
import os

#                                       Parameters
# -------------------------------------------------------------------------------------
# Parameter #1 
# Sentence Length (SL) -> Measures sentence length under syntactic complexity
def sentence_length(sentence):
    count = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            count += 1
    return count

# Helper function to get the height of a tree given its root
def tree_height(root):
    if not list(root.children):
        return 1
    else:
        return 1 + max(tree_height(x) for x in root.children)

# Parameter #2
# Height of the parse tree -> Measures structural complexity under syntactic complexity
def parse_tree_height(sentence):
    return tree_height(sentence.root)

# Parameter #3
# Type to token ratio (TTR) -> Measures lexical density under lexical complexity
def type_to_token_ratio(sentence):
    token_set = set()
    total_words = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            token_set.add(token.text.lower())
            total_words += 1    
    return round(len(token_set)/total_words,2)

# Parameter #4
# Average Word Length (AWL) -> Measures lexical sophastication under lexical complexity
def average_word_length(sentence):
    total_chars = 0
    total_words = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            total_chars += len(token.text)
            total_words += 1
    return round(total_chars/total_words,2)

# Helper function to count number of syllables in a word
def syllables_count(word): 
    return textstatistics().syllable_count(word) 

# Parameter #5
# Average Syllable Count (ASC) -> Measures lexical sophastication under lexical complexity
def average_syllable_count(sentence):
    total_syllables = 0
    total_words = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            total_syllables += syllables_count(token.text)
            total_words += 1
    return round(total_syllables/total_words,2)

# Parameter #6
# Difficult Words Ratio (DWR) -> Measures lexical sophastication under lexical complexity
def difficult_words_ratio(sentence): 
    diff_words_set = set() 
    total_words = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            syllable_count = syllables_count(token.text)
            if token.text not in easy_word_set and syllable_count >= 2: 
                diff_words_set.add(token.text) 
            total_words += 1
  
    return round(len(diff_words_set)/total_words,2) 

# Parameter #7
# Part of Speech Ratio (POS) -> Measures lexical density under lexical complexity
def pos_ratio(sentence):
    type_list = []
    token_ctr = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            token_ctr += 1
            type_list.append(token.pos_)
    
    return round(len(set(type_list))/token_ctr,2)


# Parameter #8
# Visual Words Ratio (VWR) -> Measures imagery ratings of a text sample
def visual_words_ratio(sentence):
    visual__dict = json.load(open("VReadA/visual_words.txt"))
    ctr_visual = 0
    total_words = 0
    for token in sentence:
        # Count as word only if it is not punctuation, space or a symbol
        if token.pos_ != "PUNCT" and token.pos_ != "SPACE" and token.pos_ != "SYM":
            word = token.text
            if word.upper() in visual__dict and int(visual__dict[word.upper()]) >= 300:
                ctr_visual += 1
            total_words += 1  
    return round(ctr_visual/total_words,2)

# Parameter #9
# Gunning Fog Index (GFI) -> General readability score
def gunning_fog(sentence): 
    per_diff_words = (difficult_words_ratio(sentence) * 100) + 5
    grade = 0.4 * (sentence_length(sentence) + per_diff_words) 
    return round(grade,2) 

# Parameter #10
# Flesch Kincaid Grade (FKG) -> General readability score
def flesch_kincaid_grade(sentence):
    grade = 0.39 * (sentence_length(sentence)) + 11.8 * (average_syllable_count(sentence)) - 15.59
    return round(grade,2)

# Parameter #11
# Flesch Reading Ease (FRE) -> General readability score
def flesch_reading_ease(sentence):
    ease = 206.835 - 1.015 * (sentence_length(sentence)) - 84.6 * (average_syllable_count(sentence))
    return round(ease,2)

# Parameter 12
# Similarity of the sentence with the nearby sentences -> Measures coherence of the text
def sentence_coherence(sentence,nearby_sentences):
    if len(nearby_sentences) == 0:
        # Case when there is only one sentence in the text, so fully coherent
        return 1
    
    total_score = 0
    total_pairs = 0

    for dis in nearby_sentences.keys():
        similarity = sentence.similarity(nearby_sentences[dis])

        # Give more weightage to the similarity with the sentences at
        # distance 1
        score = similarity/abs(dis)

        total_score += score
        total_pairs += 1

    return round(total_score/total_pairs,2)
# -------------------------------------------------------------------------------------
      
# Function used by outside programs to get all scores values
# of all sentences present in all the texts passed using the 
# passed nlp Language object
def get_scores_from_texts(texts,nlp,model_data):
    scores = []
    # Use piping to process the texts
    docs = nlp.pipe(texts)

    for doc in docs:
        sentences = list(doc.sents)
        number_of_sentences = len(sentences)
        for i in range(0,number_of_sentences):
            sentence = sentences[i]
            parameters_dictionary = OrderedDict()
            # For each parameter, pass the value and 0 if its low value is good ansd
            # 1 if its high value is good for readability

            parameters_dictionary["SL"] = [sentence_length(sentence),0]
            parameters_dictionary["PTH"] = [parse_tree_height(sentence),0]
            parameters_dictionary["TTR"] = [type_to_token_ratio(sentence),0]
            parameters_dictionary["AWL"] = [average_word_length(sentence),0]
            parameters_dictionary["ASC"] = [average_syllable_count(sentence),0]
            parameters_dictionary["DWR"] = [difficult_words_ratio(sentence),0]
            parameters_dictionary["POS"] = [pos_ratio(sentence),0]
            parameters_dictionary["VWR"] = [visual_words_ratio(sentence),1]
            parameters_dictionary["GFI"] = [gunning_fog(sentence),0]
            parameters_dictionary["FKG"] = [flesch_kincaid_grade(sentence),0]
            parameters_dictionary["FRE"] = [flesch_reading_ease(sentence),1]

            # Compute sentences which are less than 2 sentences away from the sentence both sides
            nearby_sentences = {}

            # Sentences at a distance of 1
            if i-1 >= 0:
                nearby_sentences[-1] = sentences[i-1]
            if i+1 < number_of_sentences:
                nearby_sentences[1] = sentences[i+1] 

            # Sentences at a distance of 2
            if i-2 >= 0:
                nearby_sentences[-2] = sentences[i-2]
            if i+2 < number_of_sentences:
                nearby_sentences[2] = sentences[i+2] 
            
            parameters_dictionary["PRP"] = [get_perplexity(sentence,model_data),0]
            parameters_dictionary["COH"] = [sentence_coherence(sentence,nearby_sentences),1]
            scores.append(parameters_dictionary)
    
    return scores


# Get the corpus stats dictionary present in the corpus folder
def get_corpus_dict(corpus_name):
    file_name = corpus_name+"/sentence_level_stats.txt"
    if not os.path.exists(file_name):
        return None
    
    corpus_dict = json.load(open(file_name))
    corpus_dict = corpus_dict["Per Sentence Scores"]
    return corpus_dict


# Calculate Z score of val using the corpus dict and code of the parameter
def calculate_z_score(val,corpus_dict,code):
    ret = (val-corpus_dict[code]["M"])/(corpus_dict[code]["STDEV"])
    if corpus_dict[code]["REV"] == 1:
        ret = ret*-1
    return ret


# Get the z scores of a text by using the stats file present in the folder corpus_name
# This is done using the passed nlp Langauge object
def get_z_scores_from_text(file_text,corpus_name,nlp,model_data):
    corpus_dict = get_corpus_dict(corpus_name)
    if corpus_dict is None:
        return None

    z_scores = []
    doc = nlp(file_text)
    sentences = list(doc.sents)
    number_of_sentences = len(sentences)
    for i in range(0,number_of_sentences):
        sentence = sentences[i]
        print(sentence)
        print("======================")
        parameters_dictionary = {}
        parameters_dictionary["SL"] = calculate_z_score(sentence_length(sentence),corpus_dict,"SL")
        parameters_dictionary["PTH"] = calculate_z_score(parse_tree_height(sentence),corpus_dict,"PTH")
        parameters_dictionary["TTR"] = calculate_z_score(type_to_token_ratio(sentence),corpus_dict,"TTR")
        parameters_dictionary["AWL"] = calculate_z_score(average_word_length(sentence),corpus_dict,"AWL")
        parameters_dictionary["ASC"] = calculate_z_score(average_syllable_count(sentence),corpus_dict,"ASC")
        parameters_dictionary["DWR"] = calculate_z_score(difficult_words_ratio(sentence),corpus_dict,"DWR")
        parameters_dictionary["POS"] = calculate_z_score(pos_ratio(sentence),corpus_dict,"POS")   
        parameters_dictionary["VWR"] = calculate_z_score(visual_words_ratio(sentence),corpus_dict,"VWR")
        parameters_dictionary["GFI"] = calculate_z_score(gunning_fog(sentence),corpus_dict,"GFI")
        parameters_dictionary["FKG"] = calculate_z_score(flesch_kincaid_grade(sentence),corpus_dict,"FKG")
        parameters_dictionary["FRE"] = calculate_z_score(flesch_reading_ease(sentence),corpus_dict,"FRE")
        parameters_dictionary["PRP"] = calculate_z_score(get_perplexity(sentence,model_data),corpus_dict,"PRP")
        # Compute sentences which are less than 2 sentences away from the sentence both sides
        nearby_sentences = {}

        # Sentences at a distance of 1
        if i-1 >= 0:
            nearby_sentences[-1] = sentences[i-1]
        if i+1 < number_of_sentences:
            nearby_sentences[1] = sentences[i+1] 

        # Sentences at a distance of 2
        if i-2 >= 0:
            nearby_sentences[-2] = sentences[i-2]
        if i+2 < number_of_sentences:
            nearby_sentences[2] = sentences[i+2] 
        
        parameters_dictionary["COH"] = calculate_z_score(sentence_coherence(sentence,nearby_sentences),corpus_dict,"COH")
        z_scores.append(parameters_dictionary)
    
    return z_scores


















