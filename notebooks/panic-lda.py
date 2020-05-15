import sys, os
import pandas as pd
import numpy as np
from textblob import TextBlob
import seaborn as sns
import matplotlib.pyplot as plt

sys.path.insert(1, '../scripts')
root = os.getcwd()

from TwintDataMiner import TwintDataMiner
from PandasDataCleaner import PandasDataCleaner
# from TextBlobSentiment import TextBlobSentiment
from Visualisation import Visualisation
from GMM import GMM
import pickle
from LDA import LDA
from pprint import pprint
from collections import Counter
import matplotlib.colors as mcolors
import argparse

parser = argparse.ArgumentParser(
    description="Parser",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--start",
    default="Cold",
    help="Hot = Load all. Warm = Load and train LDA. Cold = Complete entire data pipeline",
)

final_bow =['PPE',
 'afraid',
 'air',
 'alarmed',
 'alarming',
 'ante la crisis',
 'anxiety',
 'anxious',
 'apprehensive',
 'atmosphere',
 'brutal',
 'brutal new',
 'cascade warning heard unheeded',
 'chaos',
 'chaotic',
 'charity',
 'clean air',
 'clean water',
 'comfort',
 'coronavirus fear',
 'crise du coronavirus',
 'crisis',
 'crisis del',
 'crisis del coronavirus',
 'danger',
 'dangerous',
 'dead',
 'death',
 'disaster',
 'disastrous',
 'distress',
 'distressed',
 "don't panic",
 'donations',
 'earth',
 'el tema del coronavirus',
 'encourage',
 'encouraging',
 'environment',
 'evil',
 'exaggerate',
 'exaggerating',
 'exaggeration',
 'faint',
 'fantastic',
 'fearful',
 'fearschaos',
 'fighter',
 'fitness',
 'flabbergast',
 'flabbergasted',
 'flatten',
 'food wastage',
 'freaked',
 'free',
 'fright',
 'frightened',
 'genocide',
 'gloves',
 'government funds',
 'gratitude',
 'grief',
 'grievance',
 'gruesome',
 'happy',
 'harm',
 'healing',
 'health issue',
 'heartbreaking',
 'hell',
 'helpless',
 'honor',
 'hope',
 'horrendous',
 'horrific',
 'horrify',
 'horrifying',
 'hospital',
 'humanity',
 'improvement',
 'improving',
 'issue weakened immune',
 'job pay proposal help',
 'kill',
 'killed',
 'killingthreat',
 'kills',
 'la crise',
 'la crisis',
 'la crisis del',
 'la crisis del coronavirus',
 'life-threatening',
 'masks',
 'meditation',
 'member single family killing',
 'mild',
 'murder',
 'negative',
 'nervous',
 'not panic',
 'not serious',
 'nurses',
 'optimism',
 'optimistic',
 'outbreak cascade warning heard',
 'overcome',
 'overreact',
 'overreacted',
 'overreacting',
 'ozone',
 'panic',
 'panick',
 'panicked',
 'panicking',
 'panicky',
 'paranoid',
 'pasta',
 'phobia',
 'positive',
 'problem',
 'progress',
 'promise',
 'promising',
 'protect',
 'proud',
 'quality',
 'ran out',
 'rattle',
 'rattled',
 'recover',
 'recovery',
 'relief',
 'respect',
 'rice',
 'risk',
 'sad',
 'sadness',
 'scared',
 'scarefear',
 'scarier',
 'scariest',
 'scarily',
 'shock',
 'shocked',
 'shocking',
 'shockingly',
 'spanish flu',
 'stop aggravation',
 'strength',
 'strong',
 'stronger',
 'successfully',
 'support',
 'surreal',
 'terrible',
 'terrified threats',
 'terror',
 'tests',
 'thank you',
 'thanks',
 'threatening',
 'together',
 'toilet paper',
 'tragic',
 'travel alert',
 'trust',
 'unbearable',
 'unhygienic',
 'unnerve',
 'unnerved',
 'unnerving',
 'unserious',
 'upset',
 'upsetting',
 'upsets',
 'upsettingly',
 'vaccine',
 'warned coronavirus effect private meeting',
 'warning',
 'warning heard unheeded',
 'wonderful',
 'worried',
 'worryingly',
 'worry',
 'worst possible leader deal coronavirus',
 '⚠️',
 '😡',
 '😢',
 '😩',
 '😭',
 '😳']

#----------------------------------------------------- FUNCTIONS -------------------------------------------------------#
def cleandf(df):
    pdc = PandasDataCleaner()
    
    print("detecting language...")
    df  = pdc.detectLanguage(df, 'tweet', 'lang') 
    print("done.")
    
    print("Filtering tweets by english...")
    df = df.loc[df['lang'] == 'en']
    
    print("Cleaning...")
    df = pdc.cleanTweets(df, 'tweet')

#     df = df.reset_index(drop=True)

    return df
    

def get_pickle_object(name):
    frame = pd.read_pickle(f"{name}.pkl")
    pkl = get_full_frame(frame, str(name))
    obj = pickle.loads(pkl[0])
    return obj

def get_full_frame(frame, name):
    return list(frame[name][0:len(frame.index)])

def store_as_pickle(name, obj):
    pkl = pickle.dumps(obj)
    gmm_frame = pd.DataFrame({str(name): pkl}, index=[0])
    gmm_frame.to_pickle(f"./{name}.pkl")

def t_SNE(lda_model, corpus, type_model, topics, date, passes):
    # Get topic weights and dominant topics ------------
    from sklearn.manifold import TSNE
    from bokeh.plotting import figure, output_file, show
    from bokeh.models import Label
    from bokeh.io import output_notebook

    # Get topic weights
    topic_weights = []
    for i, row_list in enumerate(lda_model[corpus]):
        topic_weights.append([w for i, w in [row_list[0]]])

    # Array of topic weights    
    arr = pd.DataFrame(topic_weights).fillna(0).values

    # Keep the well separated points (optional)
    arr = arr[np.amax(arr, axis=1) > 0.35]

    # Dominant topic number in each doc
    topic_num = np.argmax(arr, axis=1)

    # tSNE Dimension Reduction
    tsne_model = TSNE(n_components=1, verbose=1, angle=.99, init='pca')
    # ValueError: n_components=2 must be between 1 and min(n_samples, n_features)=1 with svd_solver='randomized'

    tsne_lda = tsne_model.fit_transform(arr)

    # Plot the Topic Clusters using Bokeh
    # output_notebook()
    n_topics = len(topics)
    mycolors = np.array([color for name, color in mcolors.TABLEAU_COLORS.items()])
    # plot = plt.figure(title="t-SNE Clustering of {} LDA Topics".format(n_topics), 
    #             plot_width=900, plot_height=700)
    plt.scatter(x=tsne_lda[:,0], y=tsne_lda[:,1], color=mycolors[topic_num])
    
    if type_model == "orig":
        plt.savefig(f"../Results/{passes}-passes/orig/t-SNE/{len(topics)}-topics-{date}/png")
        plt.clf()
    elif type_model == "bow":
        plt.savefig(f"../Results/{passes}-passes/bow/t-SNE/{len(topics)}-topics-{date}/png")
        plt.clf()

def process_docs(df, lda):
    processed_docs = []
    bow_corpus = []
    
    tempdf = df.reset_index(drop=True)
    print("Index reset.")
    tempdf.sort_values("id", inplace=True) 
    print("Sorted by id.")
    # cleaning and dropping duplicate values
    tempdf.drop_duplicates(subset=["tweet"],inplace=True)
    print("Dropped duplicates.")
    tempdf = cleandf(tempdf)

    # lemmatise and tokenise
    return tempdf['tweet'].map(str).map(lda.preprocess)

def train_lda(no_topics, corpus_tfidf, lda, alpha, passes=1):

    print(f"{no_topics} topics:\n")
    lda_model_tfidf = lda.train(corpus_tfidf, no_topics, passes=passes, alpha=alpha, workers=4)
    
    for idx, topic in lda_model_tfidf.print_topics(-1):
        print('Topic: {} Word: {}'.format(idx, topic))
        print("\n")
    return lda_model_tfidf

def plot(model, docs, type_model, date, passes, alpha, height=1):
    topics = model.show_topics(formatted=False)
    data_flat = [w for w_list in docs for w in w_list]
    counter = Counter(data_flat)

    out = []
    for i, topic in topics:
        for word, weight in topic:
            out.append([word, i , weight, counter[word]])

    df = pd.DataFrame(out, columns=['word', 'topic_id', 'importance', 'word_count'])        

    # Plot Word Count and Weights of Topic Keywords
    fig, axes = plt.subplots(2, 2, figsize=(16,10), sharey=True, dpi=160)
    cols = [color for name, color in mcolors.TABLEAU_COLORS.items()]
    for i, ax in enumerate(axes.flatten()):
        ax.bar(x='word', height="word_count", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.5, alpha=0.3, label='Word Count')
        ax_twin = ax.twinx()
        ax_twin.bar(x='word', height="importance", data=df.loc[df.topic_id==i, :], color=cols[i], width=0.2, label='Weights')
        ax.set_ylabel('Word Count', color=cols[i])
        ax_twin.set_ylim(0, height); ax.set_ylim(0, 3500)
        ax.set_title('Topic: ' + str(i), color=cols[i], fontsize=16)
        ax.tick_params(axis='y', left=False)
        ax.set_xticklabels(df.loc[df.topic_id==i, 'word'], rotation=30, horizontalalignment= 'right')
        ax.legend(loc='upper left'); ax_twin.legend(loc='upper right')

    fig.tight_layout(w_pad=2)
    fig.suptitle('Word Count and Importance of Topic Keywords', fontsize=22, y=1.05)    
    if type_model == "bow":
        plt.savefig(f"../Results/{passes}-passes/bow/{alpha[0]}/{len(topics)}-topics-{date}")
        plt.clf()
    elif type_model == "orig":
        plt.savefig(f"../Results/{passes}-passes/orig/{alpha[0]}/{len(topics)}-topics-{date}")
        plt.clf()

def get_before_tweets(attributes, tdm):
    before_df = pd.DataFrame()

    # before
    for i in range(1, 16):
        if i < 10:
            path = f"../data/Final.nosync/before/tweets_march_2020_03_10_0{i}-00-00.csv"
        else:
            path = f"../data/Final.nosync/before/tweets_march_2020_03_10_{i}-00-00.csv"

        try:
            before_df = before_df.append(tdm.getPandasDataFrame(attributes, path))
        except FileNotFoundError:
            print(path)
            pass
    return before_df

def get_after_tweets(attributes, tdm):
    after_df = pd.DataFrame()

    # after
    for i in range(1, 24):
        if i < 10:
            path = f"../data/Final.nosync/after/tweets_march_2020_03_12_0{i}-00-00.csv"
        else:
            path = f"../data/Final./after/tweets_march_2020_03_12_{i}-00-00.csv"

        try:
            after_df = after_df.append(tdm.getPandasDataFrame(attributes, path))
        except FileNotFoundError:
            print(path)
            pass
    return after_df



def ldavis(lda_model, corpus):
    import pyLDAvis.gensim
    pyLDAvis.enable_notebook()
    vis = pyLDAvis.gensim.prepare(lda_model, corpus, dictionary=lda_model.dictionary)
    store_as_pickle(vis)
#----------------------------------------------------- FUNCTIONS -------------------------------------------------------#





def start(args):

    if args.start == "Cold":
        tdm = TwintDataMiner()
        attributes = ["id", "tweet", "place", "date"]

        before_df = get_before_tweets(attributes=attributes, tdm = TwintDataMiner())
        after_df = get_after_tweets(attributes=attributes, tdm = TwintDataMiner())

        print(before_df.shape)
        print(before_df.head)

        print(after_df.shape)
        print(after_df.head)

        before_df.drop_duplicates(subset='id')
        print(before_df.shape)
        print("----")
        after_df.drop_duplicates(subset='id')
        print(after_df.shape)


        # LDA declarations
        lda_before_orig = LDA()
        lda_after_orig = LDA()
        lda_before_bow = LDA()
        lda_after_bow = LDA()


        # Personalised BOW
        print("Personalised BOW")
        lemmatised_vocab = pd.DataFrame()
        lemmatised_vocab['column'] = [final_bow]
        lda_after_bow.dict_from_vocab(doc=lemmatised_vocab.iloc[0])
        lda_before_bow.dict_from_vocab(doc=lemmatised_vocab.iloc[0])
        print("BOW created")

        print("Processing docs...")
        before_docs = process_docs(before_df, lda_before_bow)
        print("Processed before docs.")
        after_docs = process_docs(after_df, lda_after_bow)
        print("Processed after docs.")

        # Original
        lda_before_orig.dict_from_vocab(doc=before_docs)
        lda_after_orig.dict_from_vocab(doc=after_docs)

        # Store before and after docs
        store_as_pickle("before_docs", before_docs)
        store_as_pickle("after_docs", after_docs)

        #Store lda objects
        store_as_pickle("lda_before_orig", lda_before_orig)
        store_as_pickle("lda_after_orig", lda_after_orig)
        store_as_pickle("lda_before_bow", lda_before_bow)
        store_as_pickle("lda_after_bow", lda_after_bow)

        #calculate corpus'
        bow_corpus_before_bow = lda_before_bow.generateBagOfWords(docs=before_docs)
        bow_corpus_after_bow = lda_after_bow.generateBagOfWords(docs=after_docs)

        bow_corpus_before_orig = lda_before_orig.generateBagOfWords(docs=before_docs)
        bow_corpus_after_orig = lda_after_orig.generateBagOfWords(docs=after_docs)

        # store corpus'
        store_as_pickle(name="bow_corpus_before_bow", obj=bow_corpus_before_bow)
        store_as_pickle(name="bow_corpus_after_bow", obj=bow_corpus_after_bow)

        store_as_pickle(name="bow_corpus_before_orig", obj=bow_corpus_before_orig)
        store_as_pickle(name="bow_corpus_after_orig", obj=bow_corpus_after_orig)

        # calculate tfidfs
        corpus_tfidf_before_orig = lda_before_orig.tf_idf(bow_corpus=bow_corpus_before_orig)
        corpus_tfidf_after_orig = lda_after_orig.tf_idf(bow_corpus=bow_corpus_after_orig)

        corpus_tfidf_before_bow = lda_before_bow.tf_idf(bow_corpus=bow_corpus_before_bow)
        corpus_tfidf_after_bow = lda_after_bow.tf_idf(bow_corpus=bow_corpus_after_bow)

        # store tfidfs
        store_as_pickle(name="corpus_tfidf_before_orig", obj=corpus_tfidf_before_orig)
        store_as_pickle(name="corpus_tfidf_after_orig", obj=corpus_tfidf_after_orig)

        store_as_pickle(name="corpus_tfidf_before_bow", obj=corpus_tfidf_before_bow)
        store_as_pickle(name="corpus_tfidf_after_bow", obj=corpus_tfidf_after_bow)

    elif args.start == "Warm":

        # get before and after docs
        before_docs = get_pickle_object(name="before_docs")
        after_docs = get_pickle_object(name="after_docs")
        print("Docs loaded")

        #get lda objects
        lda_before_orig = get_pickle_object(name="lda_before_orig")
        lda_after_orig = get_pickle_object(name="lda_after_orig")
        lda_before_bow = get_pickle_object(name="lda_before_bow")
        lda_after_bow = get_pickle_object(name="lda_after_bow")
        print("LDAs loaded")

        # get corpus'
        bow_corpus_before_bow = get_pickle_object(name="bow_corpus_before_bow")
        bow_corpus_after_bow = get_pickle_object(name="bow_corpus_after_bow")
        bow_corpus_before_orig = get_pickle_object(name="bow_corpus_before_orig")
        bow_corpus_after_orig = get_pickle_object(name="bow_corpus_after_orig")
        print("Corpus' loaded")

        # get tfidfs
        corpus_tfidf_before_orig = get_pickle_object(name="corpus_tfidf_before_orig")
        corpus_tfidf_after_orig = get_pickle_object(name="corpus_tfidf_after_orig")
        corpus_tfidf_before_bow = get_pickle_object(name="corpus_tfidf_before_bow")
        corpus_tfidf_after_bow = get_pickle_object(name="corpus_tfidf_after_bow")
        print("TFIDF' loaded\n\n")

        alpha_value = 0.025
        for i in range(1,4):
            passes = 10
            alpha_value *=2
            print(f"passes={passes}, alpha: {alpha_value}\n")
            print("--------------Original--------------")
            for no_topics in range (3, 5):
                alpha = [alpha_value] * no_topics
                print("Before")
                before_orig = train_lda(no_topics, corpus_tfidf_before_orig, lda_before_orig, passes=passes, alpha=alpha)
                topics = before_orig.show_topics(formatted=False)
                plot(model=before_orig, docs=before_docs, type_model="orig", date="before", passes=passes, height=0.02, alpha=alpha)
                # t_SNE(lda_model=before_orig, corpus=corpus_tfidf_before_orig, type_model="orig", topics=topics, date="before", passes=passes)
                print("------------  ------------")
                print("After")
                after_orig = train_lda(no_topics, corpus_tfidf_after_orig, lda_after_orig, passes=passes, alpha=alpha)
                topics = after_orig.show_topics(formatted=False)
                plot(model=after_orig, docs=after_docs, type_model="orig", date="after", passes=passes, height=0.02, alpha=alpha)
                # t_SNE(lda_model=after_orig, corpus=corpus_tfidf_after_orig, type_model="orig", topics=topics, date="after", passes=passes)
                print("\n\n")

            store_as_pickle(name=f"after-orig_passes={passes}", obj=after_orig)
            store_as_pickle(name=f"before-orig_passes={passes}", obj=before_orig)

            print("--------------BOW--------------")
            for no_topics in range (2, 5):
                alpha = [alpha_value] * no_topics
                print("Before")
                before_bow = train_lda(no_topics, corpus_tfidf_before_bow, lda_before_bow, passes=passes, alpha=alpha)
                topics = before_bow.show_topics(formatted=False)
                plot(model=before_bow, docs=before_docs, type_model="bow",  date="before", passes=passes, alpha=alpha)
                # t_SNE(lda_model=before_bow, corpus=corpus_tfidf_before_bow, type_model="bow", topics=topics, date="before", passes=passes)
                print("------------  ------------")
                print("After")
                after_bow = train_lda(no_topics, corpus_tfidf_after_bow, lda_after_bow, passes=passes, alpha=alpha)
                topics = after_bow.show_topics(formatted=False)
                plot(model=after_bow, docs=after_docs, type_model="bow", date="after", passes=passes, alpha=alpha)
                # t_SNE(lda_model=after_bow, corpus=corpus_tfidf_after_bow, type_model="bow", topics=topics, date="after", passes=passes)

                print("\n\n")

                store_as_pickle(name=f"after-bow_passes={passes}", obj=after_bow)
                store_as_pickle(name=f"before-bow_passes={passes}", obj=before_bow)


if __name__ == '__main__':
    args = parser.parse_args()
    # pr = cProfile.Profile()
    # pr.enable()
    start(args)
