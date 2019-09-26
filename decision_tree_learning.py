import pandas
from sklearn import tree
import graphviz

data_frame = pandas.read_csv('attack_data.csv')
labeled_data_frame = data_frame.copy()

labeled_data_frame['sky'] = labeled_data_frame['sky'].map({'sunny': False, 'rainy': True})
labeled_data_frame['air'] = labeled_data_frame['air'].map({'warm': False, 'cold': True})
labeled_data_frame['humid'] = labeled_data_frame['humid'].map({'normal': False, 'high': True})
labeled_data_frame['wind'] = labeled_data_frame['wind'].map({'strong': False, 'weak': True})
labeled_data_frame['water'] = labeled_data_frame['water'].map({'warm': False, 'cool': True})
labeled_data_frame['forecast'] = labeled_data_frame['forecast'].map({'same': False, 'change': True})

print(data_frame.head())
print("\n-------------------------------------------------------\n")
print(labeled_data_frame.head())

X = labeled_data_frame.drop('attack', axis=1)
y = labeled_data_frame['attack']

decision_tree = tree.DecisionTreeClassifier(criterion='entropy')
decision_tree.fit(X, y)

dot_data = tree.export_graphviz(decision_tree=decision_tree,
                                filled=True,
                                rounded=True,
                                feature_names=X.columns,
                                class_names=decision_tree.classes_
                                )

graph = graphviz.Source(dot_data)
graph.render("attack_plot")
