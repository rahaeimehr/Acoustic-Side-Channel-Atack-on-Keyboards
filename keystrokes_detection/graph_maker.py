import logging
import numpy as np
from treelib import Tree
from dictionary_checker import english_dictionary_checker

# logging.basicConfig(handlers=[
#     # logging.FileHandler("log.txt"),
#     logging.StreamHandler()
# ], level=print)


class PredictedTree:
    def __init__(self, predicted_keys, number_of_char_in_word):
        self.predicted_keys = predicted_keys
        self.tree = Tree()
        self.tree.create_node("root", "root")  # root
        self.number_of_char_in_word = number_of_char_in_word
        self.first_keys = []
        self.second_keys = []
        self.nodes = []
        self.predictions = []
        self.output_words = []
        self.remove_duplicates()
        self.tree_maker_faster()
        print(f"graph depth:{self.tree.depth()}")
        print(f"graph size:{self.tree.size()}")
        if self.tree.size() > 7000:
            raise ValueError("tree size is very big...")
        self.find_deepest_leaf_faster()
        # self.tree_show()


    def tree_show(self):
        self.tree.show()

    def from_leaf_to_root(self, nid, pred_num):
        parent_node = self.tree.parent(nid)
        if parent_node:
            # print("step:", parent_node.data)
            # print(parent_node.tag)
            # print()
            if parent_node.tag != 'root':
                self.predictions[pred_num].append(parent_node.tag)
                self.from_leaf_to_root(parent_node.identifier, pred_num)

    # def print_prediction(self):
    #     print("predicted keystrokes:")
    #     for my_list in self.predictions:
    #         print(list(reversed(my_list)))

    def find_deepest_leaf_faster(self):
        paths = self.tree.paths_to_leaves()

        for index, nodes in enumerate(paths):

            self.predictions.append([])
            if len(nodes) == self.tree.depth() + 1:
                for node_id in nodes:
                    self.predictions[index].append(self.tree.get_node(node_id).tag)
            # print("preds:", self.predictions)

    def remove_duplicates(self,):
        import itertools
        for steps_i in range(len(self.predicted_keys)):
            self.predicted_keys[steps_i].sort()
            self.predicted_keys[steps_i] = list(k for k,_ in itertools.groupby(self.predicted_keys[steps_i]))

    def tree_maker_faster(self):
        # print(f"making the tree...")
        self.nodes = [[] for _ in range(len(self.predicted_keys))]
        self.second_keys = [[] for _ in range(len(self.predicted_keys))]
        self.first_keys = [[] for _ in range(len(self.predicted_keys))]

        for steps_i in range(len(self.predicted_keys)):
            for first_key, second_key in self.predicted_keys[steps_i]:
                if steps_i == 0:
                    if first_key not in self.first_keys[steps_i]:
                        f_k = self.tree.create_node(f"{first_key}", parent="root", data=str(steps_i + 1))  # root
                        s_k = self.tree.create_node(f"{second_key}", parent=f_k.identifier,
                                                    data=str(steps_i + 2))  # root
                        self.nodes[steps_i].append([f_k, s_k])
                        self.first_keys[steps_i].append(first_key)
                        self.second_keys[steps_i].append(second_key)
                    else:
                        index_k_z = [i for i, x in enumerate(self.first_keys[steps_i]) if x == first_key]
                        # todo
                        # index_k_z = self.first_keys[steps_i].index(first_key)
                        # node_k = self.nodes[steps_i][index_k_z][0]
                        node_k = self.nodes[steps_i][index_k_z[0]][0]
                        s_k = self.tree.create_node(f"{second_key}", parent=node_k.identifier,
                                                    data=str(steps_i + 2))  # root
                        self.nodes[steps_i].append([node_k, s_k])
                        self.second_keys[steps_i].append(second_key)
                        self.first_keys[steps_i].append(first_key)

                # (step_i != 0)
                else:
                    if first_key in self.second_keys[steps_i - 1]:
                        index_k = [i for i, x in enumerate(self.second_keys[steps_i - 1]) if x == first_key]
                        for rep_index in index_k:
                            node_k = self.nodes[steps_i - 1][rep_index][1]
                            s_k = self.tree.create_node(f"{second_key}", parent=node_k.identifier,
                                                        data=str(steps_i + 2))  # root
                            self.nodes[steps_i].append([None, s_k])
                            self.second_keys[steps_i].append(second_key)
                            self.first_keys[steps_i].append(first_key)

    def print_prediction_faster(self):
        # print("predicted keystrokes:")
        all_results = []
        for my_list in self.predictions:
            if my_list:
                my_list = my_list[1:]
                # print("prediction ==>", "".join(my_list).replace("Space", " "))
                result = []
                start = 0
                for i, val in enumerate(my_list):
                    if val == 'Space':
                        if my_list[start:i]:
                            result.append(my_list[start:i])
                        start = i + 1
                if my_list[start:]:
                    result.append(my_list[start:])
                if result == []:
                    continue    
                if len(result[0]) == self.number_of_char_in_word:
                    all_results.append(result)
                    # print("Raw Prediction", "".join(result[0]))
        # print("\nDetected words after removing spaces:")
        # counter = 0
        # for res in all_results:
        #     if len(res) == 1:
        #         word = "".join(res[0])
        #         print(counter, word)
        #         counter += 1

        # print("\nDetected words after Checking in dictionary:")
        # ind = 1
        for res in all_results:
            sentence = english_dictionary_checker(res)
            if sentence: 
                # print(ind, ":", sentence)
                # ind +=1
                self.output_words.append(sentence)
        return self.output_words
    
    def my_prediction(self, num_desired_keystrokes):
        all_results = []
        for my_list in self.predictions:
            if my_list:
                my_list.pop(0)
                if len(my_list) != num_desired_keystrokes:
                    continue
                if 'Space' in my_list:
                    continue
                merged_str = "".join(my_list)
                # print(my_list)
                all_results.append(merged_str)
                
                
        return all_results

