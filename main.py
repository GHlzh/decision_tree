import operator
from math import log

# 是否回家（yes or no）
dataSet = [["a", 0, 0, 0, 'no'],
           ["a", 0, 0, 1, 'no'],
           ["a", 1, 0, 1, 'no'],
           ["a", 0, 1, 0, 'yes'],
           ["b", 0, 0, 2, 'no'],
           ["b", 0, 0, 0, 'no'],
           ["b", 0, 0, 1, 'no'],
           ["b", 1, 0, 2, 'yes'],
           ["c", 1, 1, 2, 'yes'],
           ["c", 0, 0, 2, 'yes'],
           ["c", 0, 1, 2, 'yes'],
           ["c", 1, 0, 1, 'yes'],
           ["c", 0, 0, 1, 'yes'],
           ["c", 1, 0, 2, 'yes'],
           ["c", 0, 0, 0, 'no']]
labels = ['大学所在的区域', '交通时长', '隔离时间', '隔离费']
# A区域：a，B区域：b，C区域：c
# 短：1， 长：0
# 短：1， 长：0
# 隔离费：少：0，多：1，非常多：2

# 计算香农熵
def calculate_shannon_entropy(data_set):
    data_num = len(data_set)
    # print(data_num)
    labels_num = {}
    # 统计yes和no各有几人
    for feature in data_set:
        current_label = feature[-1]
        # print(current_label)
        if current_label not in labels_num:
            labels_num[current_label] = 0
        # print(labels_num)
        labels_num[current_label] += 1
    # print(labels_num)
    shannon_entropy = 0
    for key in labels_num:
        probability = float(labels_num[key])/data_num
        shannon_entropy -= probability*log(probability, 2)
    # print(shannon_entropy)
    return shannon_entropy


# 将数据划分，根据各种特征下是“yes”还是“no”进行划分
def split_dataset(data_set, axis, value):
    ret_dataset = []
    for feat_vec in data_set:
        if feat_vec[axis] == value:
            # 从第一个特征到需要区分的特征
            reduced_feat_vec = feat_vec[:axis]
            # 跳过需要区分的特征，到最后一个特征
            reduced_feat_vec.extend(feat_vec[axis + 1:])
            ret_dataset.append(reduced_feat_vec)
    return ret_dataset


# print(split_dataset(dataSet, 4, 'yes'))
# 计算信息增益，根据结果选择最优结点
def choose_best_feature(data_set):
    # 特征数量
    total_features = len(data_set[0]) - 1
    # print(total_features)
    # 计算香农熵
    shannon_entropy = calculate_shannon_entropy(data_set)
    # 信息增益
    best_information_gain = 0.0
    # 最优特征索引值
    optimal_feature_index_value = -1
    # 遍历所有特征
    for i in range(total_features):
        # 获取数据集的第i个特征
        feature_list = [exampel[i] for exampel in data_set]
        # print(feature_list)
        # 取出每一种特征中出现的情况种类
        unique_value = set(feature_list)
        # print(unique_value)
        # 计算条件熵
        empirical_conditional_entropy = 0.0
        for value in unique_value:
            # 划分的子集
            sub_dataset = split_dataset(data_set, i, value)
            # print(sub_dataset)
            # 计算每一种选择所占比例
            prob = len(sub_dataset) / float(len(data_set))
            # print(pro)
            empirical_conditional_entropy += prob * calculate_shannon_entropy(sub_dataset)
        # 信息增益
        information_gain = shannon_entropy - empirical_conditional_entropy
        print("第%d个特征的信息增益%.3f" % (i, information_gain))
        # 更新信息增益值
        if information_gain > best_information_gain:
            best_information_gain = information_gain
            # 记录信息增益最大特征的索引值
            best_feature = i
    print("最优索引值："+str(best_feature))
    print()
    return best_feature


# choose_best_feature(dataSet)

# 以下代码为决策树的输出

# 计算class_list中每种元素的个数
def majority_cnt(class_list):
    class_count = {}
    for vote in class_list:
        if vote not in class_count:
            class_count[vote] = 0
            class_count[vote] += 1
        # 根据字典的值降序排列
        sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)
    return sorted_class_count[0][0]


def creat_tree(dataSet, labels, featLabels):
    # 取分类标签(是否回家：yes or no)
    class_list = [exampel[-1] for exampel in dataSet]
    # 如果类别完全相同则停止分类
    if class_list.count(class_list[0]) == len(class_list):
        return class_list[0]
    # 遍历完所有特征时返回出现次数最多的类标签
    if len(dataSet[0]) == 1:
        return majority_cnt(class_list)
    # 选择最优特征
    best_feature = choose_best_feature(dataSet)
    # 最优特征的标签
    best_feature_label = labels[best_feature]
    featLabels.append(best_feature_label)
    # 根据最优特征的标签生成树
    my_tree = {best_feature_label: {}}
    # 删除已使用标签
    del(labels[best_feature])
    # 得到训练集中所有最优特征的属性值
    feat_value = [exampel[best_feature] for exampel in dataSet]
    # 去掉重复属性值
    unique_vls = set(feat_value)
    for value in unique_vls:
        my_tree[best_feature_label][value] = creat_tree(split_dataset(dataSet, best_feature, value), labels, featLabels)
    return my_tree


def get_num_leaves(my_tree):
    num_leaves = 0
    first_str = next(iter(my_tree))
    second_dict = my_tree[first_str]
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':
            num_leaves += get_num_leaves(second_dict[key])
        else:
                num_leaves += 1
    return num_leaves


def get_tree_depth(my_tree):
    max_depth = 0       # 初始化决策树深度
    firsr_str = next(iter(my_tree))     # python3中myTree.keys()返回的是dict_keys,不在是list,所以不能使用myTree.keys()[0]的方法获取结点属性，可以使用list(myTree.keys())[0]
    second_dict = my_tree[firsr_str]    # 获取下一个字典
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ == 'dict':     # 测试该结点是否为字典，如果不是字典，代表此结点为叶子结点
            this_depth = 1 + get_tree_depth(second_dict[key])
        else:
            this_depth = 1
        if this_depth > max_depth:
            max_depth = this_depth      # 更新层数
    return max_depth


def classify(input_tree, feat_labels, test_vec):
    # 获取决策树节点
    first_str = next(iter(input_tree))
    # 下一个字典
    second_dict = input_tree[first_str]
    feat_index = feat_labels.index(first_str)

    for key in second_dict.keys():
        if test_vec[feat_index] == key:
            if type(second_dict[key]).__name__ == 'dict':
                class_label = classify(second_dict[key], feat_labels, test_vec)
            else:
                class_label = second_dict[key]
    return class_label


print(dataSet)
print()
print(calculate_shannon_entropy(dataSet))
print()

myTree = creat_tree(dataSet, labels, labels)
print(myTree)
