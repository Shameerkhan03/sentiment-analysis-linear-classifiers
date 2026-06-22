import project1 as p1
import utils

#-------------------------------------------------------------------------------
# Problem 7 (Part 1)
#-------------------------------------------------------------------------------

train_data = utils.load_data('reviews_train.tsv')
val_data = utils.load_data('reviews_val.tsv')
test_data = utils.load_data('reviews_test.tsv')

train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

dictionary = p1.bag_of_words(train_texts, False)

train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

#-------------------------------------------------------------------------------
# Problem 7 (Part 2)
#-------------------------------------------------------------------------------

T = 10
L = 0.01

pct_train_accuracy, pct_val_accuracy = \
   p1.classifier_accuracy(p1.perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:35} {:.4f}".format("Training accuracy for perceptron:", pct_train_accuracy))
print("{:35} {:.4f}".format("Validation accuracy for perceptron:", pct_val_accuracy))

avg_pct_train_accuracy, avg_pct_val_accuracy = \
   p1.classifier_accuracy(p1.average_perceptron, train_bow_features,val_bow_features,train_labels,val_labels,T=T)
print("{:43} {:.4f}".format("Training accuracy for average perceptron:", avg_pct_train_accuracy))
print("{:43} {:.4f}".format("Validation accuracy for average perceptron:", avg_pct_val_accuracy))

avg_peg_train_accuracy, avg_peg_val_accuracy = \
   p1.classifier_accuracy(p1.pegasos, train_bow_features,val_bow_features,train_labels,val_labels,T=T,L=L)
print("{:50} {:.4f}".format("Training accuracy for Pegasos:", avg_peg_train_accuracy))
print("{:50} {:.4f}".format("Validation accuracy for Pegasos:", avg_peg_val_accuracy))


#------------------------------------------------------------------------------
# Plot classification
#------------------------------------------------------------------------------

# def plot_classification_results(algo_name, thetas):
#     print('theta for', algo_name, 'is', ', '.join(map(str,list(thetas[0]))))
#     print('theta_0 for', algo_name, 'is', str(thetas[1]))
#     utils.plot_toy_data(algo_name, train_bow_features, train_labels, thetas)

#-------------------------------------------------------------------------------
# Practice
#-------------------------------------------------------------------------------

# train_data = utils.load_data('reviews_train.tsv')
# val_data = utils.load_data('reviews_val.tsv')
# test_data = utils.load_data('reviews_test.tsv')

# train_texts, train_labels = zip(*((sample['text'], sample['sentiment']) for sample in train_data))
# val_texts, val_labels = zip(*((sample['text'], sample['sentiment']) for sample in val_data))
# test_texts, test_labels = zip(*((sample['text'], sample['sentiment']) for sample in test_data))

# dictionary = p1.bag_of_words(train_texts, False
#                              )

# train_bow_features = p1.extract_bow_feature_vectors(train_texts, dictionary)
# val_bow_features = p1.extract_bow_feature_vectors(val_texts, dictionary)
# test_bow_features = p1.extract_bow_feature_vectors(test_texts, dictionary)

# T = 100
# L = 0.2

# thetas_perceptron = p1.perceptron(train_bow_features, train_labels, T)
# thetas_avg_perceptron = p1.average_perceptron(train_bow_features, train_labels, T)
# thetas_pegasos = p1.pegasos(train_bow_features, train_labels, T, L)

# plot_classification_results('Perceptron', thetas_perceptron)
# plot_classification_results('Average Perceptron', thetas_avg_perceptron)
# plot_classification_results('Pegasos', thetas_pegasos)