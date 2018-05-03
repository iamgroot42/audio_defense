import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix


import csv
import gensim
import numpy as np
from sklearn.svm import SVC
import scipy
from sklearn.model_selection import GridSearchCV
from gensim.models import doc2vec
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier

def read_file(filePath):
	contents = []
	with open(filePath, 'r') as f:
		for line in f:
			if ": " in line:
				contents.append(line.rstrip().split(': ')[1])
	return contents


if __name__ == "__main__":
	common_prefix = "API_results/"
	names = ["Bing_p.csv", "google_cloud_p.csv", "google_speech_p.csv", "IBM_p.csv", "sphinx_p.csv"]
	c_names = ["Bing_c.csv", "google_cloud_c.csv", "google_speech_c.csv", "IBM_c.csv", "sphinx_c.csv"]
	names = [common_prefix + x for x in names]
	c_names = [common_prefix + x for x in c_names]
	preds = [ read_file(x) for x in names]
	c_preds = [ read_file(x) for x in c_names]
	perturbed = [ [preds[j][i] for j in range(len(names))] for i in range(len(preds[0]))]
	clean= [ [c_preds[j][i] for j in range(len(names))] for i in range(len(c_preds[0]))]
	clean_old = clean[:]
	should_exit = True
	while should_exit:
		print len(clean)
		for i in range(len(clean_old)):
			clean.append(clean_old[i])
			if len(clean) == 439:
				should_exit = False
				break
	perturbed =perturbed[:439]
	print len(clean), len(perturbed)
	#exit()
	# model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
	#model = gensim.models.Word2Vec(size=150, window=10, min_count=2, workers=10)
	#model = gensim.models.Word2Vec.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)  
	#model = gensim.models.keyedvectors.KeyedVectors.load_word2vec_format('./GoogleNews-vectors-negative300.bin', binary=True)
	model = doc2vec.Doc2Vec.load("doc2vec_5dim.model")

	def anal(X):
		features = [ [model.infer_vector(x) for x in p] for p in X]
		features = np.stack(features)
		#features = features.reshape((features.shape[0] * features.shape[1], features.shape[2]))
		print features, features.shape
		#exit()
		new_features = []
		for i in range(features.shape[0]):
			this_one = []
			for j in range(5):
				for k in range(j):
					this_one.append(scipy.spatial.distance.cosine(features[i][j], features[i][k]))
			new_features.append(this_one)
		#new_features = [[ scipy.spatial.distance.cosine(x, y) for x in y] for y in features]
		new_features = np.stack(new_features)
		#print new_features.shape
		return new_features
	
	clean_anal = anal(clean)
	perturbed_anal = anal(perturbed)
	X, Y = [], []
	for i in range(len(clean_anal)):
		X.append(clean_anal[i])
		Y.append(0)
	for i in range(len(perturbed_anal)):
		X.append(perturbed_anal[i])
		Y.append(1)
	parameters = {'kernel':('linear', 'rbf'), 'C':[0.01, 0.1, 1, 10, 100]}
	#svc = SVC()
	#clf = GridSearchCV(svc, parameters)
	#clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
	clf = RandomForestClassifier(max_depth=20)
	X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.33)
	clf.fit(X_train, Y_train)
	print("Train Accuracy", clf.score(X_train, Y_train))
	print("Test Accuracy", clf.score(X_test, Y_test))
	fpr, tpr, thresholds = metrics.roc_curve(Y_test, clf.predict(X_test))
	plt.figure()
	lw = 2
	print tpr, fpr
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for our ensemble-defense against Carlini-CTC based attack (blackbox)')
	plt.legend(loc="lower right")
	plt.savefig('newROC.png')
