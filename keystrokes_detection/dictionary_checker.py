import nltk


# download the nltk English corpus if necessary
try:
    nltk.data.find('corpora/words')
except LookupError:
    nltk.download('words')  

def english_dictionary_checker(result):
    english_words = set(nltk.corpus.words.words())
    sentence = ''
    for res in result:
        merged_str = "".join(res)
        if merged_str.lower() in english_words:
            sentence += ' ' + merged_str
        else:
            sentence = ''
    if sentence:
        # print(sentence)
        with open('./prediction.txt', 'a') as f:
            f.write(sentence + '\n')
    return sentence

# import enchant
#
#
# # create an English dictionary object
# d = enchant.Dict("en_US")
#
#
# def english_dictionary_checker(result):
#     sentence_words = [dictionary_checker("".join(res)) for res in result]
#     sentence = " ".join(filter(None, sentence_words))
#     if sentence:
#         print(sentence)
#     return sentence
#
#
# def dictionary_checker(word):
#     if d.check(word):
#         return word
#     else:
#         return None
#
#
# # def english_dictionary_checker(result):
# #     sentence = ''
# #     for res in result:
# #         merged_str = "".join(res)
# #         dict_result = dictionary_checker(merged_str)
# #
# #         if dict_result:
# #             sentence = sentence + ' ' + dict_result
# #         else:
# #             sentence = ''
# #             # print(sentence)
# #     if sentence:
# #         print(sentence)
# #         with open('./prediction.txt', 'a') as f:
# #             f.write(sentence + '\n')
# #
