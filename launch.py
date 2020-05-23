from VReadA.corpus_scorer import accumulate_sentence_level_scores, get_sentence_level_scores
from VReadA.file_scorer import analyze_file

import spacy
import mxnet as mx
import gluonnlp as nlp

def printBold(text):
    BOLD = '\033[1m'
    END = '\033[0m'
    print(BOLD+text+END)


if __name__ == "__main__":
    printBold("Welcome to VreadA, the Visual Readability Analyzer!")

    print("Loading the pre-trained English Perplexity Model...")
    num_gpus = 1
    context = [mx.gpu(i) for i in range(num_gpus)] if num_gpus else [mx.cpu()]
    awd_model_name = 'awd_lstm_lm_1150'
    awd_model, vocab = nlp.model.get_model(
        awd_model_name,
        dataset_name="wikitext-2",
        pretrained=True,
        ctx=context[0],
        root="VReadA/models/")
    
    data = [awd_model,vocab,context]
    print("Loaded the pre-trained English Perplexity Model.")

    print("Loading the Spacy English Model...")

    # Load the English model for future calls
    nlp = spacy.load("en_core_web_md")
    # # Add sentencizer to the start of the pipeline
    nlp.add_pipe(nlp.create_pipe("sentencizer"),first=True) 
    # # Remove entity recognizer as not needed
    nlp.remove_pipe("ner")

    print("Loaded the Spacy English model.")

    while True:
        print("\n> ",end="")
        command = input()

        # Option used to accumulate the corpus stats initially
        if command == "score corpus":
            directory_name = input("\nEnter the corpus directory path: ")
            status,message = accumulate_sentence_level_scores(directory_name,nlp,data)
            if not status:
                print(message)
                continue

            get_sentence_level_scores(directory_name)
            continue
        
        # Option used to analyze a file 
        if command == "analyze file":
            file_name = input("\nEnter the path of the file to be analyzed: ")
            directory_name = input("\nEnter the corpus directory path: ")
            status,message = analyze_file(file_name,directory_name,nlp,data)
            if not status:
                print(message)
            continue
        
        # Option used to end the tool
        if command == "exit":
            break