import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
iris = datasets.load_iris()
X=iris.data
Y=iris.target
target_names = iris.target_names
lda = LinearDiscriminantAnalysis(n_components=2)
X_r = lda.fit_transform(X,Y)

colors = ['navy','turquoise','darkorange']
for color,i ,target_names in zip(colors,[0,1,2],target_names):
    plt.scatter(X_r[Y==i,0],X_r[Y==i,1],color=color,alpha=.8,label=target_names)
    plt.legend(loc='best',shadow=False,scatterpoints=1)
    plt.title('PCA of IRIS dataset')
    plt.show()
