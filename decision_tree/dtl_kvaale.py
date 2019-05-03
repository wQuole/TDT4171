from random import randint, choice
from math import log
from sys import argv


class DecisionNode:
    '''
    A DecisionTree consists of DecisionNodes
    '''
    def __init__(self, root):
        self.subtree = {}
        self.attribute = root

    def createSubTree(self, subtree, value):
        self.subtree[value] = subtree


def counter(examples):
    '''
    Counts the number of each type of example in the dataset
    :param examples: training data
    :return: number of class1 and class2 --> int, int
    '''
    class1, class2 = 0,0
    for example in examples:
        if example[-1] == 1:
            class1 += 1
        else:
            class2 += 1
    return class1, class2


def binary_entropy(q):
    '''
    :param q: Probability q
    :return: float between 0 and 1
    '''
    if q == 1 or q == 0:
        return 0
    return -(q * log(q) + (1 - q) * log(1 - q))


def expected_info_gain(examples, attributes):
    '''
    :param examples: training data
    :param attributes: an attribute
    :return: the best attribute --> which maximizes gain
    '''
    best_gain = -1
    best_attribute = 0
    p, n = counter(examples)
    
    for attribute in attributes:
        gain = binary_entropy(p / (p + n)) - remainder(attribute, examples)
        if gain > best_gain:
            best_gain = gain
            best_attribute = attribute

    return best_attribute


def importance(attributes, examples, func):
    '''
    :param attributes: list of attributes
    :param examples: training data
    :param func: the function it will weight the importance by
    :return: func --> random, infogain or IOError
    '''
    if func == "random":
        return choice(attributes)
    elif func == "infogain":
        return expected_info_gain(examples, attributes)
    else: return IOError("Choose 'infogain' or 'random' as func parameter ...")


def remainder(attribute, examples):
    '''
    :param attribute:  an attribute
    :param examples:  training data
    :return: the remainder
    '''
    remainder = 0
    for label in [1,2]:
        pk = 0
        nk = 0
        for example in examples:
            if example[attribute] == label:
                pk += 1
            else:
                nk += 1
        remainder += (pk + nk) / len(examples) * binary_entropy(pk /(pk + nk))
    return remainder


def equal_classification(examples):
    '''
    :param examples: training data
    :return: True or False
    '''
    class1, class2 = counter(examples)
    if (class1 == len(examples)):
        return True
    elif (class2 == len(examples)):
        return True
    return False


def plurality_value(examples):
    '''
    :param examples: training data
    :return: 1 or 2
    '''
    class1, class2 = 0,0
    for example in examples:
        if example[-1] == 1: class1 += 1
        else: class2 += 1
    if class1 == class2: return choice([1,2]) # random choice if equal
    return 1.0 if class1 > class2 else 2.0



def decision_tree_learning(examples, attributes, parent_examples, func):
    '''
    Decision Tree Learning Algorithm
    :param examples: current training examples
    :param attributes: attributes --> [Attribute 1, Attribute 2, .. Attribute 7]
    :param parent_examples: training examples from previous iteration
    :return: a tree
    '''
    if len(examples) == 0:
        return plurality_value(parent_examples)
    elif equal_classification(examples):
        return examples[0][-1]
    elif len(attributes) == 0:
        return plurality_value(examples)
    else:
        best_attribute = importance(attributes, examples, func)
        tree = DecisionNode(best_attribute)
        attributes_minus_best = [attribute for attribute in attributes if attribute != best_attribute]  
        for vk in [1,2]:
            exs = [example for example in examples if example[best_attribute] == vk]
            subtree = decision_tree_learning(exs, attributes_minus_best, examples, func)
            tree.createSubTree(subtree, vk)
        return tree


def predict(example, tree):
    '''
    :param example: test data
    :param tree: DecisionTree consisting of DecisionNodes
    :return: 1 or 2
    '''
    current_node = tree
    attribute = current_node.attribute
    if example[attribute] == 1:
         subtree = current_node.subtree[1]
         if (type(subtree) is int):
             return subtree
         return predict(example, subtree)
    else:
        subtree = current_node.subtree[2]
        if (type(subtree) is int):
            return subtree
        return predict(example, subtree)


def print_tree(tree, spacing=""):
    '''Just a printing method'''
    if type(tree) is not DecisionNode:
       print(spacing + "| --> Predicts: ", tree)
       return
    print(spacing + "| Attribute: " + str(tree.attribute))

    print(spacing + "| --> Value: 1")
    print_tree(tree.subtree[1], spacing + "|\t\t")

    print(spacing + "| --> Value: 2")
    print_tree(tree.subtree[2], spacing + "|\t\t")


def read_data(path, filename):
    '''
    :param filename: A .txt file
    :return: [[]] --> Matrix where each row represents an example
    '''
    training_data = []
    with open(path + filename, "r") as dataset:
        for row in dataset.read().splitlines():
            currentLine = row.split("\t")
            training_data.append([int(attr) for attr in currentLine])
        dataset.close()
        return training_data


# M a i n
def main():
    # Choose between "infogain" and "random"
    imprtnc = argv[1].lower()
    
    attributes = range(0,7) # [0, 1, 2, 3, 4, 5, 6]
    
    path = "/Users/wquole/PycharmProjects/TDT4171/decision_tree/dataset/"

    # Datasets
    training_data = read_data(path, 'training.txt')
    testing_data = read_data(path, 'test.txt')

    # Build Tree
    tree = decision_tree_learning(training_data, attributes, training_data, func=imprtnc)
    #for values in tree.subtree.values():
    #    print("value",values)
    print_tree(tree)

    # Testing
    matches = 0
    print("\n\nTrained on {} examples\t-->\ttesting on {} examples\n".format(len(training_data), len(testing_data)))
    for example in testing_data:
        pred = predict(example, tree)
        print("Actual: {}\t |\t Predicted: {}\t |\t Hit: {}".format(example[-1], pred, (bool(example[-1]==pred))))
        if pred == example[-1]:
            matches += 1

    accuracy = matches/len(testing_data)
    missed = len(testing_data) - matches
    print("\nMissed:\t\t{}\nMatches:\t{}\nAccuracy:\t{} %\n\nUsing the '{}' as importance function".format(missed, matches, accuracy*100, imprtnc))



if __name__ == '__main__':
    main()

