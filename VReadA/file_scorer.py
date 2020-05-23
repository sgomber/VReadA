from VReadA.scorer import get_scores_from_texts,get_z_scores_from_text
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import json
import csv
import sys
import re
import os


# plot and store the obtained z values in the result file
def plot_and_store_z(data,filename):
    score_labels = []
    for k in data[0].keys():
        score_labels.append(k)

    file_data = []
    for d in data:
        sent_data = []
        for k in d.keys():
            sent_data.append(d[k])
        file_data.append(sent_data)
    file_data = np.asarray(file_data)

    y_labels = [x+1 for x in range(0,len(file_data))]

    fig = plt.figure(constrained_layout=True)
    fig.set_size_inches(10,13)

    fig.suptitle('Sentence-wise z-score visualization') 
    fig.subplots_adjust(top=0.95)

    gs = fig.add_gridspec(2,16)
    ax1 = fig.add_subplot(gs[0,0:2])
    ax2 = fig.add_subplot(gs[0,3:9])
    ax3 = fig.add_subplot(gs[0,10:13])
    ax4 = fig.add_subplot(gs[0,14:16])
    ax5 = fig.add_subplot(gs[1,0:13])
    ax6 = fig.add_subplot(gs[1,14:16])

    chart1 = sns.heatmap(file_data[:,0:2], 
                         ax=ax1, 
                         xticklabels=score_labels[0:2], 
                         linewidths=0.5, 
                         cmap="Blues",
                         vmax=5,
                         vmin=-5,
                         cbar=False)
    chart1.set_xlabel("Syntactic")
    chart1.set_yticklabels(labels=y_labels)
    chart1.set_ylabel("Sentence number")
    
    chart2 = sns.heatmap(file_data[:,2:8], 
                         ax=ax2, 
                         xticklabels=score_labels[2:8], 
                         linewidths=0.5, 
                         cmap="Blues", 
                         vmax=5,
                         vmin=-5,
                         cbar=False)
    chart2.set_yticklabels(labels=y_labels)
    chart2.set_xlabel("Lexical")
    chart2.set_yticklabels(labels=y_labels)

    chart3 = sns.heatmap(file_data[:,8:11], 
                         ax=ax3, 
                         xticklabels=score_labels[8:11], 
                         linewidths=0.5, 
                         cmap="Blues",
                         vmax=5,
                         vmin=-5,
                         cbar=False)
    chart3.set_xlabel("General")
    chart3.set_yticklabels(labels=y_labels)

    chart4 = sns.heatmap(file_data[:,11:13], 
                         ax=ax4, 
                         xticklabels=score_labels[11:13], 
                         linewidths=0.5, 
                         cmap="Blues",
                         vmax=5,
                         vmin=-5,
                         cbar_kws={'label': 'Normalized Z-Scores'})
    chart4.set_xlabel("ML Based")
    chart4.set_yticklabels(labels=y_labels)

    p = {}
    for l in range(0,13):
        if l<=1:
            p[l]="lightcoral"
        elif l>1 and l<=7:
            p[l]="lightgreen"
        elif l>7 and l<=10:
            p[l]="gold"
        else:
            p[l]="plum"

    chart5 = sns.boxplot(data=file_data,ax=ax5,palette=p)
    yextent = ax5.get_ylim()
    ax5.set(ylim=(min(-5,yextent[0]),max(5,yextent[1])))
    chart5.set_xticklabels(labels=score_labels)
    chart5.set_ylabel("Normalized z-values")
    chart5.set_title("Distribution of Parameters")
    
    custom_lines = [Line2D([0], [0], color="lightcoral", lw=2),
                Line2D([0], [0], color="lightgreen", lw=2),
                Line2D([0], [0], color="gold", lw=2),
                Line2D([0],[0], color="plum",lw=2)]
    ax5.legend(custom_lines, ['Syntactic', 'Lexical', 'General Readability','ML Based'])

    means = []
    for c in range(0,13):
        means.append(np.mean(file_data[:,c]))
    means = np.asarray(means)
    mean_sent = np.mean(means[0:2])
    mean_lex = np.mean(means[2:8])
    mean_gen = np.mean(means[8:11])
    mean_adv = np.mean(means[11:13])

    data_tot = [[mean_sent],[mean_lex],[mean_gen],[mean_adv]]
    d = np.asarray(data_tot)
    chart6 = sns.heatmap(d, 
                        ax=ax6, 
                        linewidths=0.5, 
                        cmap="Blues",
                        vmax=3,
                        vmin=-3,
                        cbar_kws={'label': 'Normalized Z-Scores'})
    chart6.set_yticks([0.5,1.5,2.5,3.5])
    chart6.set_yticklabels(labels=['Syntactic','Lexical','General','ML'],va="center")
    chart6.set_title(label="Type-wise Summary",fontdict={"fontsize":10})
    chart6.set_xticklabels(["Complete Text"])


    plt.savefig(filename)
    plt.show()

def store_scores(data,filename):
    with open(filename,"w") as csvfile:
        csvwriter = csv.writer(csvfile)

        header = []
        for k in data[0].keys():
            header.append(k)
        csvwriter.writerow(header)

        for d in data:
            scores = []
            for k in d.keys():
                scores.append(d[k][0])
            csvwriter.writerow(scores)

# Get the path where to store the output plot
def get_outputplot_filename(input_file_name):
    return "Results/Results_" + os.path.basename(input_file_name).split(".")[0] + ".png"

# Get the path where to store the output parameter scores
def get_outputscores_filename(input_file_name):
    return "Results/Results_" + os.path.basename(input_file_name).split(".")[0] + ".csv"

def analyze_file(filename,corpus,nlp,model_data):
    input_file_name = filename

    # Check if file is present
    if not os.path.exists(input_file_name):
        return False, input_file_name + " is not a valid file."

    corpus_folder = corpus
    # Check if corpus directory is present
    if not os.path.isdir(corpus_folder):
        return False, corpus_folder + " is not a valid directory."

    text = open(input_file_name,"r").read()
    # Reduce multispaces to a single space
    text = re.sub(' +', ' ',text)
    # Remove new line characters
    text = re.sub('\n',' ',text)
    file_text = [text]
    # Get the scores of the text
    scores = get_scores_from_texts(file_text,nlp,model_data)

    # Get Z values
    z_values = get_z_scores_from_text(file_text[0],corpus_folder,nlp,model_data)
    open("data.txt","w").write(json.dumps(z_values))

    if z_values is None:
        return False, "The passed corpus does not contain the stats file!"

    if not os.path.exists('Results'):
        os.makedirs('Results')
    output_scores_file = get_outputscores_filename(input_file_name)
    output_plot_file = get_outputplot_filename(input_file_name)
    store_scores(scores,output_scores_file)
    plot_and_store_z(z_values,output_plot_file)

    print("Results calculated and stored in " + output_scores_file + " and " + output_plot_file + " in the Results folder.")
    return True,"OK"
