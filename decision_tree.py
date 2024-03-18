import heapq
import math

class Node:
    def __init__(self, dataset, point_estimate, split_feature, ig):
        self.dataset = dataset
        self.point_estimate = point_estimate
        self.split_feature = split_feature
        self.info_gain = ig
        self.left = None
        self.right = None
    def is_leaf(self):
        return self.left is None and self.right is None

class PriorityQueue:
    def __init__(self):
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop_highest_priority(self):
        if self._queue:
            return heapq.heappop(self._queue)[-1]
        else:
            raise IndexError("pop from an empty priority queue")

pq = PriorityQueue()
def buildTree(ds, info_gain_fun, limit):
    # at each step, we need to calculate the expected information gain and pick
    # the one with the most information gain
    ig, split_feature = get_next_best_feature(ds, info_gain_fun)
    point_estimate = get_point_estimate(ds)

    first_node = Node(ds, point_estimate, split_feature, ig)
    tree = first_node
    
    pq.push(first_node, ig)

    internal_nodes = 1

    #print("start ", first_node.split_feature, first_node.info_gain, len(first_node.dataset), first_node.point_estimate)
    while internal_nodes <= limit: #(100 internal nodes)
        # get node of highest piority (IE)
        node = pq.pop_highest_priority()
        #print("node feature", node.split_feature, node.info_gain)

        # split data at leaf 
        feature = node.split_feature
        data_set = node.dataset
        left_set, right_set = split_ds(data_set, feature)

        # compute point estimates
        left_pe = get_point_estimate(left_set)
        right_pe = get_point_estimate(right_set)


        # compute next best feature to split on and information gain        
        l_ig, left_feature = get_next_best_feature(left_set, info_gain_fun)
        r_ig, right_feature = get_next_best_feature(right_set, info_gain_fun)


        # create node structures 
        left_node = Node(left_set, left_pe, left_feature, l_ig)
        right_node = Node(right_set, right_pe, right_feature, r_ig)
        
        # set children of parent node
        node.left = left_node
        node.right = right_node

        #print("l c : ", left_node.split_feature, left_node.info_gain, len(left_node.dataset), left_node.point_estimate)
        #print("r c : ", right_node.split_feature, right_node.info_gain, len(right_node.dataset), right_node.point_estimate)

        # push to pq
        pq.push(right_node, r_ig)
        pq.push(left_node, l_ig)

        # increase intneral nodes
        internal_nodes +=1
        
        # here we can also test the accuracy

        if info_gain_fun == "weighted":
            train_weighted_acur_array.append(get_acc(tree, train_data, doc_id_label))
            test_weighted_acur_array.append(get_acc(tree, test_data, test_doc_id_label))
        else:
            train_unweighted_acur_array.append(get_acc(tree, train_data, doc_id_label))
            test_unweighted_acur_array.append(get_acc(tree, test_data, test_doc_id_label))    
            
    return tree
    
def get_point_estimate(ds):
    # if half is one thing then its that thing
    if get_result_counts(ds, 1)/len(ds) > 0.5:
        return 1
    return 2

def get_next_best_feature(ds, info_gain_method):
    # returns word, and information gain
    unique_words = set()
    for d in ds.values():
        unique_words.update(d)

    unique_words = sorted(unique_words)

    max_ig = 0
    feature = ''

    for word in unique_words:
        ig = 0
        if info_gain_method == "weighted":
            ig = fraction_doc_info_gain(ds, word)
        else:
            ig = avg_info_gain(ds,word)
        if ig > max_ig:
            max_ig = ig
            feature = word
    
    return max_ig, feature
        
def get_result_counts(ds, cond):
    # maps the result ds and counts number of results a or b in ds
    count = 0
    for key in ds:
        if doc_id_label[key] == cond:
            count +=1
    return count

def get_entropy(ds):
    counts = {1: 0, 2: 0}
    #print("counts ", counts)
    for doc in ds:
        counts[doc_id_label[doc]] +=1
    
    if 0 in counts.values():
        return 0

    entropy = 0.0
    total_instances = len(ds)

    for item in counts:
        prob = counts[item] / total_instances
        entropy -= prob * math.log2(prob)

    return entropy


def fraction_doc_info_gain(ds, word):
    # IG = I(E) - [N1/N * I(E1) + N2/N * I(E2)]
    pior_entropy = get_entropy(ds)
    have_ds, not_have_ds = split_ds(ds, word)

    # P(have) and P(not have)
    p_have = len(have_ds) / len(ds)
    p_not_have = 1 - p_have

    # get IE1 and IE2
    I_E1 = get_entropy(have_ds)
    I_E2 = get_entropy(not_have_ds)

    # I(E Split) 
    split = (p_have * I_E1) + (p_not_have * I_E2)

    # IE
    IG = pior_entropy - split
    return IG

def avg_info_gain(ds,word):
    # IG = I(E) - [1/2 * I(E1) + 1/2 * I(E2)]
    pior_entropy = get_entropy(ds)

    have_ds, not_have_ds = split_ds(ds, word)

    # get IE1 and IE2
    I_E1 = get_entropy(have_ds)
    I_E2 = get_entropy(not_have_ds)

    # I(E Split) 
    split = (1/2 * I_E1) + (1/2 * I_E2)

    # IE
    IG = pior_entropy - split
    return IG

words = {}
# id : word

with open ("data/words.txt") as fileIn:
    word_id = 1
    for word in fileIn:
        words[word_id] = word.strip('\n')
        word_id += 1
#print(words)

doc_id_label = {}
test_doc_id_label = {}
train_data = {}
test_data = {}
with open ("data/trainLabel.txt") as fileIn:
    doc_id = 1
    for label in fileIn:
        doc_id_label[doc_id] = int(label.strip('\n'))
        train_data[doc_id] = []
        doc_id += 1

with open ("data/testLabel.txt") as fileIn:
    doc_id = 1
    for label in fileIn:
        test_doc_id_label[doc_id] = int(label.strip('\n'))
        test_data[doc_id] = []
        doc_id += 1

#docid: [word1, word2, .., ]
with open ("data/trainData.txt") as fileIn:
    for line in fileIn:
        line = line.strip('\n').split(" ")
        doc_id = int(line[0])
        word_id = int(line[1])
        train_data[doc_id].append(words[word_id])

with open ("data/testData.txt") as fileIn:
    for line in fileIn:
        line = line.strip('\n').split(" ")
        doc_id = int(line[0])
        word_id = int(line[1])
        test_data[doc_id].append(words[word_id])


def split_ds(ds, word):
    # returns 2 datasets
    # - dataset of documents with word in it
    # - dataset of documents without word in it
    d1 = {}
    d2 = {}
    for key in ds:
        if word not in ds[key]:
            d2[key] = ds[key]
        else:
            d1[key] = ds[key]
    return d1, d2

def get_acc(tree, ds, doc_label):
    # returns total number of correct/len(ds) * 100
    correct = 0
    for doc in ds:
        result = use_dt(tree,ds[doc])
        if result == -1:
            print("ERROR")
            pass
        if result == doc_label[doc]:
            correct +=1
    
    return correct / len(ds) * 100
        
def use_dt(tree, doc):
    # using the decision tree, determine
    # the subreddit of this document
    cur_node = tree
    while cur_node:
        if cur_node.is_leaf():
            # if we are the leaf of the decision tree,
            # we want to return the result
            return cur_node.point_estimate

        if cur_node.split_feature in doc:
            # if the item is in doc, we go left
            cur_node = cur_node.left
        else:
            # if the item is not in the doc, we go right
            cur_node = cur_node.right
    return -1

train_weighted_acur_array = []
test_weighted_acur_array = []
train_unweighted_acur_array = []
test_unweighted_acur_array = []

weighted_tree = buildTree(train_data, "weighted", 100)
unweighted_tree = buildTree(train_data, "unweighted", 100)

print(train_weighted_acur_array,
test_weighted_acur_array ,
train_unweighted_acur_array,
test_unweighted_acur_array )

with open ("train_weighted_classifier_accuracy2.txt", "w") as file:
    for item in train_weighted_acur_array:
        file.write(str(item) + '\n') 
        
with open ("test_weighted_classifier_accuracy2.txt", "w") as file:
    for item in test_weighted_acur_array:
        file.write(str(item) + '\n')

with open ("train_unweighted_classifier_accuracy2.txt", "w") as file:
    for item in train_unweighted_acur_array:
        file.write(str(item) + '\n') 
        
with open ("test_unweighted_classifier_accuracy2.txt", "w") as file:
    for item in test_unweighted_acur_array:
        file.write(str(item) + '\n')
