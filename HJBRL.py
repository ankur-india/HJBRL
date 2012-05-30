# -*- coding: utf-8 -*-

# Imports #######################################################################

import nltk
import re
import codecs
import itertools as itx

from nltk.tokenize.api import *

#################################################################################





# Variables #####################################################################

bangla_alphabet = dict(
    consonant         = u'[\u0995-\u09b9\u09ce\u09dc-\u09df]',
    independent_vowel = u'[\u0985-\u0994]',
    dependent_vowel   = u'[\u09be-\u09cc\u09d7]',
    dependent_sign    = u'[\u0981-\u0983\u09cd]',
    virama            = u'[\u09cd]'
)

bangla_word_pattern = re.compile(ur'''(?:
    {consonant}
    (?:{virama}{consonant})?
    (?:{virama}{consonant})?
    {dependent_vowel}?
    {dependent_sign}?
    |
    {independent_vowel}
    {dependent_sign}?
)+$'''.format(**bangla_alphabet), re.VERBOSE)

dictionary_path     = '/home/AtriyaSen/nltk_data/Dictionaries'
bdictionary_path    = '/home/AtriyaSen/nltk_data/EnglishDictionaries'
cleaned_corpus_path = '/home/AtriyaSen/nltk_data/Corpuses'
#dirty_corpus_path   = '/home/AtriyaSen/nltk_data/FIRE/bn_ABP'
dirty_corpus_path   = '/home/AtriyaSen/nltk_data/ABPBhojonAll'

result_stage1_path  = '/home/AtriyaSen/stage1.utf8'
result_stage2_path  = '/home/AtriyaSen/stage2.utf8'
result_stage3_path  = '/home/AtriyaSen/stage3.utf8'
result_stage4_path  = '/home/AtriyaSen/stage4.utf8'
result_stage5_path  = '/home/AtriyaSen/stage5.utf8'
result_residue_path = '/home/AtriyaSen/residue.utf8'

frequency_cutoff = 5

dirty_words_distribution = None

dictionary = nltk.corpus.reader.WordListCorpusReader(
    dictionary_path, '.*', encoding='utf-8'
    )
dictionary_words = set(dictionary.words())

noun_classifier_suffixes = [ur"টা$", ur"টি$", ur"থানা$", ur"থানি$", ur"জন$", ur"টুকু$", ur"গুলো$", ur"গুলি$", ur"রা$"]
noun_case_marker_suffixes = [ur"রা$", ur"দের$", u"rকে$", u"rতে$", ur"ে$"]
noun_emphasizing_suffixes = [ur"ই$", ur"ও$"]
verb_first_suffixes = [ur"ব$", ur"তাম$", ur"ি.নি$", ur"ে.ছিলাম$", ur"ছিলাম$", ur"লাম$", ur"ে.ছি$", ur"ছি$", ur"ি$"]
verb_second_suffixes = [ur"বে$", ur"তে$", ur"নি$", ur"ে.ছিলে$", ur"ছিলে$", ur"লে$", ur"ে.ছ$", ur"ছ$"]
verb_third_suffixes = [ur"বে$", ur"ত$", ur"ে.নি$", ur"ে.ছিল$", ur"ছিল$", ur"ল$", ur"ে.ছে$", ur"ছে$", ur"ে$"]

#################################################################################





# Classes #######################################################################

class BanglaWordTokenizer(StringTokenizer):
    def tokenize(self, s):
        return BanglaWordTokenizerFunction.tokenize(s)

#################################################################################





# Functions #####################################################################

BanglaWordTokenizerFunction = nltk.tokenize.RegexpTokenizer(ur'[\u0980-\u09DF]+')

def BanglaCorpusWordTokenize(corpus):
    return [w for filename in corpus.fileids() for w in corpus.words(filename)]

def WriteToFile (raw_words, filename):
    f = codecs.open(filename, 'w', 'utf-8')
    for w in dirty_words_distribution.keys():
        if w not in raw_words:
            continue
        w_count = dirty_words_distribution[w]
        f.write(w + ' (' + str(w_count) + ')\n')
    f.close()

def FilterByDictionary(raw_words):
    filtered = raw_words - dictionary_words
    print "Dictionary Filter (", len(raw_words), "->", len(filtered), ")"
    WriteToFile(raw_words & dictionary_words, result_stage1_path)
    return filtered

def FilterByBorrowedDictionary(raw_words):
    bdictionary = nltk.corpus.reader.WordListCorpusReader(
        bdictionary_path, '.*', encoding='utf-8'
        )
    bdictionary_words = set(bdictionary.words())
    filtered = raw_words - bdictionary_words
    print "Borrowed Dictionary Filter (", len(raw_words), "->", len(filtered), ")"
    WriteToFile(raw_words & bdictionary_words, result_stage2_path)
    return filtered

def FilterByCleanedCorpus(raw_words):
    cleaned_corpus =  nltk.corpus.reader.plaintext.PlaintextCorpusReader(
        cleaned_corpus_path, '.*', word_tokenizer = BanglaWordTokenizer(), encoding='utf-8'
        )
    cleaned_corpus_words = set(BanglaCorpusWordTokenize(cleaned_corpus))
    filtered = raw_words - cleaned_corpus_words
    print "Cleaned Corpus Filter (", len(raw_words), "->", len(filtered), ")"
    WriteToFile(raw_words & cleaned_corpus_words, result_stage3_path)
    return filtered

def FilterByBanglaWordPattern(raw_words):
    filtered = set()
    invalid = set()
    for w in raw_words:
        if bangla_word_pattern.match(w):
            filtered.add(w)
        else:
            invalid.add(w)
    print "Bangla Word Pattern Filter (", len(raw_words), "->", len(filtered), ")"
    WriteToFile(invalid, result_stage4_path)
    return filtered

def FilterByFrequency(raw_words):
    filtered = set()
    invalid = set()
    for w in raw_words:
        w_count = dirty_words_distribution[w]
        if w_count < frequency_cutoff:
            filtered.add(w)
        else:
            invalid.add(w)
    print "Word Frequency Filter (", len(raw_words), "->", len(filtered), ")"
    WriteToFile(invalid, result_stage5_path)
    return filtered

#########################################

def QuickStemNounByDictionary(word):
    ra_blocker_seen = 0

    if word in dictionary_words:
        return word

    for s in noun_emphasizing_suffixes:
        # Possibility of using the NLTK regex stemmer here instead
        if re.search(s, word):
            word = word[:-(len(s) - 1)]
            break

    if word in dictionary_words:
        return word
    
    for s in noun_case_marker_suffixes:
        if re.search(s, word):
            if s is "রা" or s is "কে" or s is "ে" or s is "তে":
                ra_blocker_seen = 1
            word = word[:-(len(s) - 1)]
            break
    
    if word in dictionary_words:
        return word

    for s in noun_classifier_suffixes:
        if re.search(s, word):
            if not(s is "রা" and ra_blocker_seen is 1):
                word = word[:-(len(s) - 1)]
    
    if word in dictionary_words:
        return word
    else:
        return None

def QuickStemVerbByDictionary(word):
    
    if word in dictionary_words:
        return word

    for s in verb_first_suffixes:
        # Possibility of using the NLTK regex stemmer here instead
        if re.search(s, word):
            word = word[:-(len(s) - 1)]
            break

    if word in dictionary_words:
        return word
    
    for s in verb_second_suffixes:
        if re.search(s, word):
            word = word[:-(len(s) - 1)]
            break
    
    if word in dictionary_words:
        return word

    for s in verb_third_suffixes:
        if re.search(s, word):
            word = word[:-(len(s) - 1)]
    
    if word in dictionary_words:
        return word
    else:
        return None

#################################################################################





# Main ##########################################################################

dirty_corpus = nltk.corpus.reader.plaintext.PlaintextCorpusReader(
    dirty_corpus_path, '.*', word_tokenizer = BanglaWordTokenizer(), encoding='utf-8'
    )

dirty_corpus_words_list = BanglaCorpusWordTokenize(dirty_corpus)
#dirty_corpus_words_dict = dict(it.izip(reversed(dirty_corpus_words_list), reversed(xrange(len(dirty_corpus_words_list)))))
dirty_corpus_words      = set(dirty_corpus_words_list)

#dirty_words_distribution = nltk.FreqDist(dirty_corpus_words_list)

# WriteToFile(
#     FilterByFrequency(
#         FilterByBanglaWordPattern(
#             FilterByCleanedCorpus(
#                 FilterByBorrowedDictionary(
#                     FilterByDictionary(dirty_corpus_words)
#                     )
#                 )
#             )
#         )
#     , result_residue_path)

for w in dirty_corpus_words:
    
    wStem1 = QuickStemNounByDictionary(w)
    wStem2 = QuickStemVerbByDictionary(w)
    
    if wStem1 is None and wStem2 is None:
        continue
    
    if wStem1 is None:
        if wStem2 is not w:
            print w, "->", wStem2, "(V)"
        continue
    
    if wStem2 is None:
        if wStem1 is not w:
            print w, "->", wStem1, "(N)"
        continue
    
    if len(wStem1) < len(wStem2):
        if wStem1 is not w:
            print w, "->", wStem1, "(N)"
    elif len(wStem2) < len(wStem1):
        if wStem2 is not w:
            print w, "->", wStem2, "(V)"
    else:
        if wStem1 is wStem2:
            if wStem1 is not w:
                print w, "->", wStem1, "(NV)"
        else:
            if wStem1 is not w:
                print w, "->", wStem1, "(Nmult)"
            if wStem2 is not w:
                print w, "->", wStem2, "(Vmult)"
            break

#################################################################################
