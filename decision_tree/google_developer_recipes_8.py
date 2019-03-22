'''
Following the Machine Learning Recipes from the Google Developers channel.
Here I'll write a Decision Tree Classifier
'''
dataset_src = "/Users/wquole/PycharmProjects/TDT4171/decision_tree/dataset/"


def read_data(filename):
    training_data = []
    with open(dataset_src + filename, "r") as dataset:
        for row in dataset.read().splitlines():
            currentLine = row.split("\t")
            training_data.append(list(currentLine))

        return training_data


# Heaader for printing
#header = ["color", "diameter", "label"]
header = ["attribute 1", "attribute 2", "attribute 3", "attribute 4", "attribute 5", "attribute 6", "class"]
class Question:
    '''A Question is used to partition a dataset'''
    def __init__(self, column, value):
        self.column = column
        self.value = value

    def match(self, example):
        '''
        Compare the feature value in an example
        to an value in this question
        '''
        val = example[self.column]
        if is_numeric(val):
            return val >= self.value
        else:
            return val == self.value

    def __repr__(self):
        '''Helper method for printing'''
        condition = "=="
        if is_numeric(self.value):
            condition = ">="
        return  "Is {} {} {}?".format(header[self.column], condition, str(self.value))

class Leaf:
    '''
    A Leaf Node classifies data
    Holds a dictionary of class --> number of times it appears in the rows
    from thee training data that reach this leaf
    '''
    def __init__(self, rows):
        self.predictions = class_counts(rows)

class Decision_Node:
    '''
    A Decision Node asks a question
    Holds a reference to the question, and the two child nodes
    '''
    def __init__(self, question, true_branch, false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def unique_vals(rows, col):
    '''Find the unique values for a col in a dataset'''
    return set([row[col] for row in rows])


def is_numeric(value):
    '''Check if a value is numeric'''
    return isinstance(value, int) or isinstance(value, float)


def partition(rows, question):
    '''
    Partition the dataset
    ForEach Row check if matches question
    '''
    true_rows, false_rows = [], []
    for row in rows:
        if question.match(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def gini(rows):
    '''Calculate the Gini Impurity for a list of rows'''
    counts = class_counts(rows)
    impurity = 1
    for lbl in counts:
        prob_of_lbl = counts[lbl]/float(len(rows))
        impurity -= prob_of_lbl**2  
    return impurity


def info_gain(left, right, current_uncertainty):
    '''Returns information gain'''
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * gini(left) - (1 - p) * gini(right)


def class_counts(rows):
    '''Counts the number of each type of example in a dataset'''
    counts = {}
    i = 0
    for row in rows:
        label = row[-1]
        i += 1
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def find_best_split(rows):
    '''Find the best question to ask'''
    best_gain = 0
    best_question = None
    current_uncertainty = gini(rows)
    n_features = len(rows[0]) - 1 # number of columns

    for col in range(n_features):
        values = set([row[col] for row in  rows])
        for val in values:
             question = Question(col, val)
             true_rows, false_rows = partition(rows, question)

             if (len(true_rows) == 0 or len(false_rows) == 0):
                 continue # Skip the split if it does not divide the dataset

             gain = info_gain(true_rows, false_rows, current_uncertainty)

             if gain >= best_gain: # can use '>' instead of '>='
                 best_gain, best_question = gain, question

    return best_gain, best_question


def classify(row, node):
    '''Classifying'''

    if isinstance(node, Leaf):
        #print("Node preddictions:", node.predictions)
        return node.predictions

    if node.question.match(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def build_tree(rows):
    '''Builds the tree'''

    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)
    true_branch = build_tree(true_rows)

    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(node, spacing=""):
    '''Just a printer method'''

    if isinstance(node, Leaf):
        print(spacing + "Predict", node.predictions)
        return

    print(spacing + str(node.question))

    print(spacing + "--> True")
    print_tree(node.true_branch, spacing + " ")

    print(spacing + "--> False")
    print_tree(node.false_branch, spacing + " ")



def print_leaf(counts):
    '''Leaf printer method'''
    total = sum(counts.values())
    probs = {}
    for lbl, val in counts.items():
        #print("Label:\t",lbl,"Value:\t",val)
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs



def getAccuracy(tree, testdata):
    matches = 0
    missed = 0
    for row in testdata:
        predictions = print_leaf(classify(row, tree))
        prediction = int(max(list(predictions.values())).strip("%"))/100
        if float(row[-1]) == prediction:
            matches += 1
        else:
            missed += 1
    print("\nMissed:\t\t{}\nMatches:\t{}".format(missed, matches))
    return matches/(missed+matches)




def main():
    training_data = read_data('training.txt')
    testing_data = read_data('test.txt')
    my_tree = build_tree(training_data)
    print_tree(my_tree)

    for row in testing_data:
        #print("\nRow\t\t\t",row)
        print("Actual: {}\t |\t Predicted: {}".format(row[-1], print_leaf(classify(row, my_tree))))
    print("Precision:\t{}%".format((getAccuracy(my_tree,  testing_data))*100))


main()


