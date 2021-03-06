import os
import shutil
import csv
import string
import nltk
from timeit import default_timer as timer
from time import sleep as wait
# import threading  # will potentially use multi-threading
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from nltk.stem.wordnet import WordNetLemmatizer
import scipy
from scipy.stats import moment
import numpy
nltk.download('averaged_perceptron_tagger')

#stopWords = set(stopwords.words('english'))


###############################################################################
########## function(s) for generating parse trees & document objects ##########
###############################################################################


def generate_document(text, author):

    """ function: generate_document
        ---------------------------
        extract class labels & tokenized (and sanitized) title/body text

        :param reuter_txt: parsetree of 'reuter' child in original parsetree
        :returns: dictionary representing fields of single document entity
    """
    document = init_document()
    populate_author(document, author)
    populate_wordlist(document, text)

    # UNCOMMENT WHEN DEBUGGING
    # print(document)
    # wait(10)

    return document


def parse_documents(type='train'):
    """ function: parse_document
        ------------------------
        extract list of Document objects from token list
        :returns: list of document entities generated by generate_document()
    """

    if 'train' in type:
        _dir_ = 'C50train'
    elif 'test' in type:
        _dir_ = 'C50test'
    else:
        raise Exception

    documents = []
    # generate well-formatted document set for each file
    for subdir in os.listdir(_dir_):
        # print(subdir)
        for file in (os.listdir(os.path.join(os.getcwd(), _dir_, subdir))):
            # print(file)
            f = open(os.path.join(os.getcwd(), _dir_, subdir, file), 'r')
            data = f.read()
            f.close()
            # print(data.lower())
            author = subdir
            data = data.lower()
            document = generate_document(data, author)
            documents.append(document)

        print(f"Finished extracting information from {_dir_}/ {subdir}")

    return documents


###############################################################################
################ function(s) for generating document objects ##################
###############################################################################


def init_document():
    """ function: init_document
        -----------------------
        initialize new empty document skeleton

        :returns: dictionary @document of document fields
        @dictionary['words'] is a dictionary
        @dictionary['body'] is a list for the body text terms
    """
    document = {
                # 'topics': [],
                'author': '',
                'body': ''
                }

    return document


def populate_class_label(document, article):
    """ function: populate_class_label
        ------------------------------
        extract topics from @article and fill @document

        :param document: formatted dictionary object representing a document
        :param article:  formatted parse tree built from unformatted data
        @article is a 'reuter' child of the original file parsetree
    """

    for topic in article.topics.children:
        document['topics'].append(topic.text.encode('ascii', 'ignore'))


def populate_author(document, author):
    """ function: populate_class_label
        ------------------------------
        extract author from @article and fill @document

        :param document: formatted dictionary object representing a document
        :param article:  formatted parse tree built from unformatted data
        @article is a 'reuter' child of the original file parsetree
    """
    if author:
        document['author'] = author


def populate_wordlist(document, text):
    """ function: populate_word_list
        ----------------------------
        extract title/body words from @article, preprocess, and fill @document

        :param document: formatted dictionary object representing a document
        :param article:  formatted parse tree built from unformatted data
            @article is a 'reuter' child of the original file parsetree
    """
    if text:
        document['body'] = text


#####################################################################
################## NLTK #############################################
#####################################################################


def tokenize2(text):
    """ function: tokenize
        ------------------
        generate list of tokens given a block of @text;

        :param text: string representing text field (title or body)
        :returns: list of strings of tokenized & sanitized words
    """
    # encode unicode to string
    ascii = text.encode('ascii', 'ignore')
    # remove digits
    no_digits = ascii.translate(None, string.digits)
    # remove punctuation
    no_punctuation = no_digits.translate(None, string.punctuation)
    # tokenize
    tokens = nltk.word_tokenize(no_punctuation)
    # remove stopwords - assume 'reuter'/'reuters' are also irrelevant
    no_stop_words = [w for w in tokens if not w in stopwords.words('english')]
    # filter out non-english words
    eng = [y for y in no_stop_words if wordnet.synsets(y)]
    # lemmatization process
    lemmas = []
    lmtzr = WordNetLemmatizer()
    for token in eng:
        lemmas.append(lmtzr.lemmatize(token))
    # stemming process
    stems = []
    stemmer = PorterStemmer()
    for token in lemmas:
        stems.append(stemmer.stem(token).encode('ascii','ignore'))
    # remove short stems
    terms = [x for x in stems if len(x) >= 4]

    return terms


def tokenize(text):
    for sent in nltk.sent_tokenize(text.lower()):
        for word in nltk.word_tokenize(sent):
            yield word


###############################################################################

def create_db_authors(docs):
    # najdeme autorov
    db_authors = [doc['author'] for doc in docs]

    # for doc in docs:
    #     if doc['author']:
    #         db_authors.append(doc['author'])

    db_authors = list(set(db_authors))
    # print('db_authors')
    # print(len(db_authors))

    return db_authors

#######################################################################################

def tokenize_doc(text):
    # Raw document body text
    # convert text to lower case
    text = text.lower()
    wordlist = text.replace(',', ' ').replace(',\n', ' ').replace(' "', ' ').replace('" ', ' ').replace('"', ' ').replace(',"', ' ')\
                        .replace('. ', ' ').replace('.\n', ' ').replace('(', ' ').replace(')', ' ')\
                        .replace('>', ' ').replace('<', ' ').replace(':', ' ').split()


    # Replace numbers and math signs with FLAGS
    for word in wordlist:
        # if word.isdigit():
        #     wordlist[wordlist.index(word)] = "__cislo_int__"  # word is int number
        if is_num(word):
            wordlist[wordlist.index(word)] = "_num_"  # word is 140,000 number
        elif is_digit(word):
            wordlist[wordlist.index(word)] = "_float_"  # word is float number
        # elif is_math_sign(word):
        #     wordlist[wordlist.index(word)] = "__mops__"   # word is mathematical operational sign
        elif is_webpage(word):
            wordlist[wordlist.index(word)] = "_web_"   # word is mathematical operational sign
        elif is_price(word):
            wordlist[wordlist.index(word)] = "_price_"   # word is mathematical operational sign
        else:
            # just word
            pass

    return wordlist


def is_num(x):
    rex = re.compile("^[0-9,]*$")
    return rex.fullmatch(x)


def is_digit(x):
    try:
       float(x)
       return True
    except ValueError:
        return False


def is_math_sign(x):
        if x == "/" or x == "*" or x == "+" or x == "-" or x == "=" or x == "%":
            return True
        else:
            return False


def is_webpage(word):
        if 'http:' in word or '.com' in word:
            return True
        else:
            return False


def is_price(word):
        if '$' in word:
            return True
        else:
            return False


def remove_stopwords(wordlist):
        stopWords = ['i', 'me', 'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours', 'yourself', 'yourselves',
                         'he', 'him', 'his', 'himself', 'she', 'her', 'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                         'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'this', 'that', 'these', 'those', 'am', 'is', 'are', 'was',
                         'were', 'be', 'been', 'being', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing', 'a', 'an', 'the',
                         'and', 'but', 'if', 'or', 'because', 'as', 'until', 'while', 'of', 'at', 'by', 'for', 'with', 'about', 'against',
                         'between', 'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from', 'up', 'down', 'in',
                         'out', 'on', 'off', 'over', 'under', 'again', 'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                         'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only',
                         'own', 'same', 'so', 'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don', 'should', 'now', 'd', 'll',
                         'm', 'o', 're', 've', 'y', 'ain', 'aren', 'couldn', 'didn', 'doesn', 'hadn', 'hasn', 'haven', 'isn', 'ma',
                         'mightn', 'mustn', 'needn', 'shan', 'shouldn', 'wasn', 'weren', 'won', 'wouldn']

        wordlist = [word for word in wordlist if word not in stopWords]

        return wordlist

#####################################################################################################

def create_dictionary(traindocs, testdocs, keep_percent):
    # create dict from all words in all docs and return 70% of most frequent
    docs = []
    docs.extend(traindocs)
    docs.extend(testdocs)

    # Create wordlist from docs
    wordlist = []
    for d in range(len(docs)):
        wordlist.extend(tokenize_doc(docs[d]['body']))  # tokenize and gorup numbers, prices, webpages under keywords
    print("Slovnik wordlist 1/2 DONE!")

    # Remove stop words
    wordlist = remove_stopwords(wordlist)

    # Create word frequencies
    wordcounts = nltk.FreqDist(w for w in wordlist)
    wanted_count = round((wordcounts.N()/100) * keep_percent)  # get X percent of frequency distribution
    summ = 0
    count = 0
    for i in list(wordcounts.most_common()): # zoradene podla od najcastejsie vyskyt. sa slov
        summ += i[1]  # scitavam hodnoty v stlpcoch histogramu
        count += 1    # pocitam stlpce, cez ktore prechadzam
        if summ >= wanted_count:  # dosiahnem chceny pocet slov (keep_percent), scitavanim slov postupne po sltpcoch
            break

    print(count)  # pocet roznych slov (pocet stlpcov v histograme), ktore su v ramci X percent najviac sa vyskytujucich sa slov
    print(summ)  # pocetnosti vsetkych slov spolu
    wordcounts.plot(count, title='FreqDist Histogram', cumulative=False) # 100

    print("Slovnik 2/2 DONE!")
    return wordcounts.most_common(count)

##############################################################################################

def word_frequencies(wordlist, dictionary):
    wordcounts = []
    for item in dictionary:  # dict v tvare list setov [('the', 30393), ('to', 14265), ...]
        if item[0] in wordlist:  # ak je slovo v texte
            # spocitam vyskyt daneho slova vo wordliste (dokumente)
            count = wordlist.count(item[0])
            # vyskyt daneho slova vo wordliste v percentach = frekvencia
            wordcounts.append((count / len(wordlist)) * 100)
        else:
            # __OOV__, slovo zo slovnika v texte nie je
            wordcounts.append(0.)

    return wordcounts


def sentence_lenghts(text):
    sent_lens = []
    sentences = nltk.sent_tokenize(text)  # this gives us a list of sentences
    for s in sentences:
        tokenized_sentence = nltk.word_tokenize(s)
        sent_lens.append(len(tokenized_sentence))
        # tagged = nltk.pos_tag(tokenized_text)
        # print("DEBUG: ")
        # print(sent_lens)
        # print(nltk.word_tokenize(sentences[1]))

    return sent_lens


def pos_tagging(text):
    tagged_text = []
    sentences = nltk.sent_tokenize(text)  # this gives us a list of sentences
    for s in sentences:
        tokenized_sentence = nltk.word_tokenize(s)
        tagged = nltk.pos_tag(tokenized_sentence)
        tagged_text.append(tagged)
        print("DEBUG: ")
        print(tagged)

    return tagged_text


def coll_dict(docs, num, window_size):
    # red wine is a collocation, whereas the wine is not
    # collocations are frequent bigrams/..., except that we want to pay more attention to the cases that involve rare words.
    # In particular, we want to find bigrams that occur more often than we would expect based on the frequency of the individual words.
    coll = []
    all_text = ''
    for doc in docs:
        all_text += doc['body']
    text = nltk.Text(tkn for tkn in tokenize(all_text))

    # TODO vstavanu metodu collocations som upravil riadkom ! return colloc_strings !, po preinstalovani kniznice musis UPRAVIT
    # Print collocations derived from the text, ignoring stopwords.
    coll.extend(text.collocations(num=num, window_size=window_size))
    assert len(coll) == num, "Pocet vytvorenych collocations sa nerovna ziadanemu, niekde je chyba."

    print("coll_dict DONE!")
    return coll


def often_bigrams_count(text, colloc_dict):
    fv = []
    for c in colloc_dict:  # list of tuples
        if str(c[0] + ' ' + c[1]) in text:  # z tuple na string, hladame ci je kolokacia v texte
            bigrams = list(nltk.bigrams(tokenize(text)))  # rozoberiem text na BIGRAMY
            # print(bigrams)
            count = bigrams.count(c)  # zratam kolko krat sa nachadza kolokacia v texte
            fv.append(count)
        else:
            fv.append(0)
    # print("FV: ")
    # print(fv)


    return fv

#######################################################################################
##########  feature extraction method 1 - ALL IN ONE #########################
#######################################################################################
def feature_extraction_1(docs, authors, dict):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']
        wordlist = tokenize_doc(text)
        sentences = nltk.tokenize.sent_tokenize(text)

        # lenght of sentences in text
        list_len_sentences = [len(nltk.tokenize.word_tokenize(s)) for s in sentences]

        # Word frequencies
        word_freqs = word_frequencies(wordlist, dict)
        # print("Word frequencies: ")
        # print(word_freqs)

        # Mean lenght of sentences
        mean_sentence_len = scipy.mean(list_len_sentences)
        mean_sentence_len = round(mean_sentence_len, ndigits=4)
        # print("Priemerna dlzka viet: ")
        # print(mean_sentence_len)

        # Number of stopwords in document
        wordlist_wo_sw = remove_stopwords(wordlist)
        stopwords_count = len(wordlist) - len(wordlist_wo_sw)
        # print("Pocet stop slov: ")
        # print(stopwords_count)

        # Number of all words in document
        wordlist_len = len(wordlist)
        # print("Pocet all slov: ")
        # print(wordlist_len)
        # print(wordlist)

        # Standard deviation of sentence length
        std = numpy.std(list_len_sentences)
        # print("Standard deviation: ")
        # print(std)

        # Number of different words
        vocab_richness = nltk.FreqDist(w for w in wordlist).B()
        # print(" Vocabulary richness: ")
        # print(vocab_richness)

        # Number of different words
        num_hapaxes = len(nltk.FreqDist(w for w in wordlist).hapaxes())
        #print(" Pocet slov, ktore pouzil autor len raz: ")
        #print(num_hapaxes)

        # MOMENTS
       # m1 = moment(a=list_len_sentences, moment=1)  # always 0
        m2 = moment(a=list_len_sentences, moment=2)  # variance je 2 mocnina std
        m3 = moment(a=list_len_sentences, moment=3)  # skewness - sikmost
        m4 = moment(a=list_len_sentences, moment=4)  # kurtosis - spicatost
        m5 = moment(a=list_len_sentences, moment=5)  # -
        m6 = moment(a=list_len_sentences, moment=6)  # -
        m7 = moment(a=list_len_sentences, moment=7)  # -
        m8 = moment(a=list_len_sentences, moment=8)  # -
        m9 = moment(a=list_len_sentences, moment=9)  # -
        m10 = moment(a=list_len_sentences, moment=10)  # -

        # lenght of words in text
        list_len_words = []
        for s in sentences:
            for word in nltk.tokenize.word_tokenize(s):
                list_len_words.append(len(list(word)))

        mw2 = moment(a=list_len_words, moment=2)  # variance je 2 mocnina std
        mw3 = moment(a=list_len_words, moment=3)  # skewness - sikmost
        mw4 = moment(a=list_len_words, moment=4)  # kurtosis - spicatost
        mw5 = moment(a=list_len_words, moment=5)  # -
        mw6 = moment(a=list_len_words, moment=6)  # -
        mw7 = moment(a=list_len_words, moment=7)  # -
        mw8 = moment(a=list_len_words, moment=8)  # -
        mw9 = moment(a=list_len_words, moment=9)  # -
        mw10 = moment(a=list_len_words, moment=10)  # -

        # FV dokumentu
        fv.extend(word_freqs)
        fv.append(mean_sentence_len)
        fv.append(stopwords_count)
        fv.append(wordlist_len)
        fv.append(std)
        fv.append(vocab_richness)
        fv.append(num_hapaxes)

        fv.append(m2)
        fv.append(m3)
        fv.append(m4)
        fv.append(m5)
        fv.append(m6)
        fv.append(m7)
        fv.append(m8)
        fv.append(m9)
        fv.append(m10)

        fv.append(mw2)
        fv.append(mw3)
        fv.append(mw4)
        fv.append(mw5)
        fv.append(mw6)
        fv.append(mw7)
        fv.append(mw8)
        fv.append(mw9)
        fv.append(mw10)

        #  LV dokumentu
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method 1: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

#######################################################################################
##########  feature extraction method 2  #########################
#######################################################################################
def feature_extraction_2(docs, authors):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']
        sentences = nltk.tokenize.sent_tokenize(text)

        # lenght of sentences in text
        list_len_sentences = [len(nltk.tokenize.word_tokenize(s)) for s in sentences]

        # Mean lenght of sentences
        mean_sentence_len = scipy.mean(list_len_sentences)
        mean_sentence_len = round(mean_sentence_len, ndigits=6)
        # print("Priemerna dlzka viet: ")
        # print(mean_sentence_len)

        # Number of stopwords in document
        wordlist = tokenize_doc(text)
        wordlist_wo_sw = remove_stopwords(wordlist)
        stopwords_count = len(wordlist) - len(wordlist_wo_sw)
        # print("Pocet stop slov: ")
        # print(stopwords_count)

        # Number of all words in document
        wordlist = tokenize_doc(text)
        wordlist_len = len(wordlist)
        # print("Pocet all slov: ")
        # print(wordlist_len)

        # Standard deviation
        std = round(numpy.std(list_len_sentences), ndigits=6)
        # print("Standard deviation: ")
        # print(std)

        # MOMENTS
        # m1 = moment(a=list_len_sentences, moment=1)  # always 0
        m2 = round(moment(a=list_len_sentences, moment=2), ndigits=6)  # variance je 2 mocnina std
        m3 = round(moment(a=list_len_sentences, moment=3), ndigits=6) # skewness - sikmost
        m4 = round(moment(a=list_len_sentences, moment=4), ndigits=6)  # kurtosis - spicatost
        m5 = round(moment(a=list_len_sentences, moment=5), ndigits=6)  # -
        m6 = round(moment(a=list_len_sentences, moment=6), ndigits=6)  # -
        m7 = round(moment(a=list_len_sentences, moment=7), ndigits=6)  # -
        m8 = round(moment(a=list_len_sentences, moment=8), ndigits=6)  # -
        m9 = round(moment(a=list_len_sentences, moment=9), ndigits=6)  # -
        m10 = round(moment(a=list_len_sentences, moment=10), ndigits=6)  # -

        # lenght of words in text
        list_len_words = []
        for s in sentences:
            for word in nltk.tokenize.word_tokenize(s):
                list_len_words.append(len(list(word)))

        mw2 = round(moment(a=list_len_words, moment=2), ndigits=6) # variance je 2 mocnina std
        mw3 = round(moment(a=list_len_words, moment=3), ndigits=6)  # skewness - sikmost
        mw4 = round(moment(a=list_len_words, moment=4), ndigits=6)  # kurtosis - spicatost
        mw5 = round(moment(a=list_len_words, moment=5), ndigits=6)  # -
        mw6 = round(moment(a=list_len_words, moment=6), ndigits=6)  # -
        mw7 = round(moment(a=list_len_words, moment=7), ndigits=6)  # -
        mw8 = round(moment(a=list_len_words, moment=8), ndigits=6)  # -
        mw9 = round(moment(a=list_len_words, moment=9), ndigits=6)  # -
        mw10 = round(moment(a=list_len_words, moment=10), ndigits=6)  # -

        # Number of different words
        vocab_richness = nltk.FreqDist(w for w in wordlist).B()
        # print(" Vocabulary richness: ")
        # print(vocab_richness)

        # Number of different words
        num_hapaxes = len(nltk.FreqDist(w for w in wordlist).hapaxes())
        # print(" Pocet slov, ktore pouzil autor len raz: ")
        # print(num_hapaxes)

        # FV dokumentu
        fv.append(mean_sentence_len)
        fv.append(stopwords_count)
        fv.append(wordlist_len)
        fv.append(std)
        fv.append(vocab_richness)
        fv.append(num_hapaxes)

        fv.append(m2)
        fv.append(m3)
        fv.append(m4)
        fv.append(m5)
        fv.append(m6)
        fv.append(m7)
        fv.append(m8)
        fv.append(m9)
        fv.append(m10)

        fv.append(mw2)
        fv.append(mw3)
        fv.append(mw4)
        fv.append(mw5)
        fv.append(mw6)
        fv.append(mw7)
        fv.append(mw8)
        fv.append(mw9)
        fv.append(mw10)

        #  LV dokumentu
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method 2: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

###################################################################################
##########  feature extraction method 3   MOMENTS           #########################
###################################################################################
def feature_extraction_3(docs, authors):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']
        sentences = nltk.tokenize.sent_tokenize(text)

        # lenght of words in text
        list_len_words = []
        for s in sentences:
            for word in nltk.tokenize.word_tokenize(s):
                list_len_words.append(len(word))

        # lenght of sentences in text
        list_len_sentences = [len(nltk.tokenize.word_tokenize(s)) for s in sentences]

        # m1 = moment(a=list_len_sentences, moment=1)  # always 0
        m2 = moment(a=list_len_sentences, moment=2)  # variance je 2 mocnina std
        m3 = moment(a=list_len_sentences, moment=3)  # skewness - sikmost
        m4 = moment(a=list_len_sentences, moment=4)  # kurtosis - spicatost
        m5 = moment(a=list_len_sentences, moment=5)  #  -
        m6 = moment(a=list_len_sentences, moment=6)  #  -
        m7 = moment(a=list_len_sentences, moment=7)  #  -
        m8 = moment(a=list_len_sentences, moment=8)  #  -
        m9 = moment(a=list_len_sentences, moment=9)  #  -
        m10 = moment(a=list_len_sentences, moment=10)  #  -

        mw2 = moment(a=list_len_words, moment=2)  # variance je 2 mocnina std
        mw3 = moment(a=list_len_words, moment=3)  # skewness - sikmost
        mw4 = moment(a=list_len_words, moment=4)  # kurtosis - spicatost
        mw5 = moment(a=list_len_words, moment=5)  #  -
        mw6 = moment(a=list_len_words, moment=6)  #  -
        mw7 = moment(a=list_len_words, moment=7)  #  -
        mw8 = moment(a=list_len_words, moment=8)  #  -
        mw9 = moment(a=list_len_words, moment=9)  #  -
        mw10 = moment(a=list_len_words, moment=10)  #  -

        # FV dokumentu
        fv.append(m2)
        fv.append(m3)
        fv.append(m4)
        fv.append(m5)
        fv.append(m6)
        fv.append(m7)
        fv.append(m8)
        fv.append(m9)
        fv.append(m10)

        fv.append(mw2)
        fv.append(mw3)
        fv.append(mw4)
        fv.append(mw5)
        fv.append(mw6)
        fv.append(mw7)
        fv.append(mw8)
        fv.append(mw9)
        fv.append(mw10)

        #  LV dokumentu
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method 3: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

#######################################################################################
##########  feature extraction method 4 - Word Freqs  #########################
#######################################################################################
def feature_extraction_4(docs, authors, dict):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']
        wordlist = tokenize_doc(text)
        wordlist = remove_stopwords(wordlist)

        # Word frequencies
        word_freqs = word_frequencies(wordlist, dict)
        # print("Word frequencies: ")
        # print(word_freqs)

        #  FV dokumentu
        fv.extend(word_freqs)

        #  LV dokumentu
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method 4: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

#######################################################################################
##########  feature extraction method   5   ##########################################
#######################################################################################

def feature_extraction_5(docs, authors, colloc_dict):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']

        # counts of OFTEN BIGRAMS in text, return the same dictionary but with counts on place of index of bigram in dictionary
        bigrams = often_bigrams_count(text, colloc_dict)

        # vytvarame FV
        fv.extend(bigrams)

        #  vytvarame LV
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

#######################################################################################
##########  feature extraction method   6 - HISTOGRAM DLZKY VIET    ###################
#######################################################################################

def feature_extraction_6(docs, authors):
    fvs = []
    for doc in docs:
        fv = []
        text = doc['body']

        # HISTOGRAM dlzok viet v clanku [19, 36, 34, 19, 30, 21, 46, 13, 26, 28, 37, 7, 16, 14, 21]
        # os historgramu [ < 5, 5-10, 10-15, 15-20, 20-25, 25-30, 35-40, 40 < ]
        histogram, osx = numpy.histogram(sentence_lenghts(text), range=None, normed=False, weights=None, density=None)

        # vytvarame FV
        fv.extend(histogram)

        #  vytvarame LV
        lv = authors.index(doc['author'])
        fv.append(lv)

        # FV + LV vsetkych dokumentov v liste
        fvs.append(fv)

    print("Feature extraction method: DONE !")
    return fvs   # all vectors (Feature Vectors + Labels) as list

#######################################################################################
##########  generate .csv with Input vector / Feature Vector  #########################
#######################################################################################

def create_cvs(dataset, authors, filename='train_data_fv.csv'):
    categories = authors

    try:
        shutil.rmtree(f'/{filename}', ignore_errors=False)
    except FileNotFoundError:
        pass

    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)

       # HLAVICKA
        writer.writerow([len(dataset), len(dataset[0]) - 1, len(categories), categories])# pocet riadkov (1 riadok = 1 in/out vector = 1 dokument),
                                                                                         # pocet vlastnosti = dlzka 1 FV (-1 co je LABEL),
                                                                                         # vypisane vsetky kategorie
        # DATA
        writer.writerows(dataset)     # v riadkoch su Feature data + posledny stlpec v riadku je label

    print(f"{filename} file was created.\n")


###############################################################################
################## main function - single point of execution ##################
###############################################################################

def main():
    # The training corpus consists of 2,500 texts (50 per author)
    # and the test corpus includes other 2,500 texts (50 per author)
    # non-overlapping with the training texts.

    print('\n Generating document objects...')
    train = parse_documents('train')
    test = parse_documents('test')

    print('\n Generating list of authors...')
    train_authors = create_db_authors(train)
    test_authors = create_db_authors(test)

    assert len(train_authors) == len(test_authors), "PROBLEM: pocet test a train autorov sa musi rovnat."

##################################################
###       GENERATE DICTIONARIES                ###
##################################################

    print("\n Generating frequent words dictionary...")
    diction = create_dictionary(train, test, keep_percent=70)

    print("\n Generating collocation dictionary...")
    colloc_dict = coll_dict(train+test, num=1779, window_size=2)

##############################################
###       GENERATE DATASETS                ###
##############################################

    filename_train = 'train_data_fv_bigrams_70.csv'  # MUST BE CHANGED !
    filename_test = 'test_data_fv_bigrams_70.csv'  # MUST BE CHANGED !


    ### TRAIN FV ###
    print('\nGenerating train dataset. This may take some time...')
    trainset = feature_extraction_1(train, train_authors, diction)  # all in one (2,3,4)
    # trainset = feature_extraction_2(train, train_authors)   # priemerna dlzka viet + pocet stopslov + pocet all slov + std dlzky viet + 10 MOMENTOV dlzok viet + 10 MOMENTOV dlzok slov + hapaxes + richness
    # trainset = feature_extraction_3(train, train_authors)   # 10 MOMENTOV dlzok viet + 10 MOMENTOV dlzok slov
    trainset = feature_extraction_4(train, train_authors, diction)  # only the most frequent single words- no stop words, 70%/1779, 75%/2413,
    trainset = feature_extraction_5(train, train_authors, colloc_dict)   # the most frequent bigrams - no stop words
    # trainset = feature_extraction_6(train, train_authors)   # histogram dlzok viet v hraniciach

    ### TEST FV ###
    print('\nGenerating test dataset. This may take some time...')
    # testset = feature_extraction_1(test, test_authors, diction)
    # testset = feature_extraction_2(test, test_authors)
    # testset = feature_extraction_3(test, test_authors)
    # testset = feature_extraction_4(test, test_authors, diction)
    testset = feature_extraction_5(test, test_authors, colloc_dict)
    # testset = feature_extraction_6(test, test_authors)

    assert len(testset[0]) == len(trainset[0]), "PROBLÉM: dĺžka testovacieho a trénovacieho FV sa nerovná."



    print('Generating train CSV. This may take some time...')
    create_cvs(trainset, train_authors, filename=filename_train)

    print('Generating test CSV. This may take some time...')
    create_cvs(testset, test_authors, filename=filename_test)


if __name__ == "__main__":
    main()