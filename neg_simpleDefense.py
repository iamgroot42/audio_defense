import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix

# Fix seed
#np.random.seed(69)

def readFile(filePath, type, ratio=0.2):
	X = []
	Y = []
	count = 0
	with open(filePath) as f:
		for line in f:
			if 1:
				#print line
				x = line.rstrip('\n').split(' ')
				if x[0]!='ERROR:':
					#print x[0]
					x  = [int(i) for i in x]
					X.append(x)
					Y.append(type)
			count+=1
	X, Y = np.stack(X), np.stack(Y)
	p = np.random.permutation(len(X))
	sp = int(len(p) * ratio)
	Xtr, Ytr = X[p[sp:]], Y[p[sp:]]
	Xte, Yte = X[p[:sp]], Y[p[:sp]]
	return (Xtr, Ytr), (Xte, Yte)


def combineBoth(clean, perturbed):
	(Xtr1, Ytr1), (Xte1, Yte1) = readFile(clean, 0)
	print "clean done"
	(Xtr2, Ytr2), (Xte2, Yte2) = readFile(perturbed, 1)
	minBoth = min(len(Ytr1), len(Ytr2))
	p = np.random.permutation(len(Ytr1))[:minBoth]
	Xtr1, Ytr1 = Xtr1[p], Ytr1[p]
	p = np.random.permutation(len(Ytr2))[:minBoth]
	Xtr2, Ytr2 = Xtr2[p], Ytr2[p]
	#minBoth = min(len(Yte1), len(Yte2))
	#p = np.random.permutation(len(Yte1))[:minBoth]
	#Xte1, Yte1 = Xte1[p], Yte1[p]
	#p = np.random.permutation(len(Yte2))[:minBoth]
	#Xte2, Yte2 = Xte2[p], Yte2[p]
	Xtr = np.concatenate((Xtr1, Xtr2), axis=0)
	Ytr = np.concatenate((Ytr1, Ytr2), axis=0)
	Xte = np.concatenate((Xte1, Xte2), axis=0)
	Yte = np.concatenate((Yte1, Yte2), axis=0)
	return (Xtr, Ytr), (Xte, Yte)


if __name__ == "__main__":
	(Xtr, Ytr), (Xte, Yte) = combineBoth("./neg_zero.txt", "./neg_one.txt")
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svc = SVC()
	clf = GridSearchCV(svc, parameters)
	clf.fit(Xtr, Ytr)
	print("Training accuracy", clf.score(Xtr, Ytr))
	print("Testing accuracy", clf.score(Xte, Yte))
	# Print Confusion Matrix
	cnf_matrix = confusion_matrix(Yte, clf.predict(Xte))
	print(cnf_matrix)
	# Plot ROC
	fpr, tpr, thresholds = metrics.roc_curve(Yte, clf.predict(Xte))
	plt.figure()
	lw = 2
	plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % metrics.auc(fpr, tpr))
	plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ROC curve for our defense against Carlini-CTC based attack')
	plt.legend(loc="lower right")
	plt.savefig('neg_simpleROC.png')
	
