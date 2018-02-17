from sklearn import svm
import numpy as np
import data
import matplotlib.pyplot as plt

class KSVMWrap:
    def __init__(self, X, Y_, param_svm_c=1, param_svm_gamma='auto'):
        """ Konstruira omotač i uči RBF SVM klasifikator
            X,Y_:            podatci i točni indeksi razreda
            param_svm_c:     relativni značaj podatkovne cijene
            param_svm_gamma: širina RBF jezgre
        """
        self.clf = svm.SVC(C=param_svm_c, gamma=param_svm_gamma)
        self.clf.fit(X, Y_)

    def predict(self, X):
        return self.clf.predict(X)

    def get_scores(self, X):
        return self.clf.decision_function(X)

    def support(self):
        return self.clf.support_

if __name__ == "__main__":
    np.random.seed(100)
    X, Y_ = data.sample_gmm(6, 2, 10)
    svm = KSVMWrap(X, Y_)
    Y = svm.predict(X)

    # ispiši performansu (preciznost i odziv po razredima)
    accuracy, recall, precision = data.eval_perf_binary(Y, Y_)
    print("Accuracy: {}".format(accuracy))
    print("Preciznost: {}".format(precision))
    print("Odziv: {}".format(recall))

    # iscrtaj rezultate, decizijsku plohu
    bbox=(np.min(X, axis=0), np.max(X, axis=0))
    data.graph_surface(svm.get_scores, bbox, offset=0)
    data.graph_data(X, Y_, Y, special=svm.support())
    plt.show()
    