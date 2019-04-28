from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import scipy.io
import DPalgs
import util
import numpy as np
import csv
import pickle
import input
from scipy.io import loadmat
import tensorflow as tf

FLAGS = tf.flags.FLAGS

tf.flags.DEFINE_string('dataset', 'adult', 'The name of the dataset to use')
tf.flags.DEFINE_integer('nb_labels', 10, 'Number of output classes')

tf.flags.DEFINE_string('data_dir','/Users/yuqing/github_proj/optimal_dp_linear_regression/data','Temporary storage')

path = '/Users/yuqing/Desktop/uci/wine/wine.mat'
csv_path =  '/Users/yuqing/Desktop/uci/wine/winequality-red.csv'
mat_path = '/Users/yuqing/github_proj/optimal_dp_linear_regression/data/bike/bike.mat'
mat = scipy.io.loadmat(path)




def normalize(X):
    Y = X/(np.sum(X*X,axis=1)**0.5)[:,None]
    return Y



def train(func,X,y,eps,delta):
    output = func(X,y,eps,delta)
    if type(output) is tuple:
        theta = output[0]
    else:
        theta = output
    return theta

def predict(theta,X):
    return np.dot(X,theta)

def eval_mse(y,yhat):
    return np.mean((y-yhat)**2)


def init(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        data = list(reader)

    for i in range(len(data)):
        data[i] = data[i][0].split()
        data[i] = [float(t) for t in data[i]]

    data = np.array(data)
    X = data[:, 0:-1]
    Y = data[:, -1]
    y = [int(i) for i in Y]
    y = Y
    X = normalize(X)
    X = np.array(X)
    y = np.array(y)
   # y = Y/np.max(Y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    return  X_train, X_test, y_train, y_test


def train_log( X,y, eps,delta):

    """
    : param a: start search point of a
    :param X:
    :param y:
    :param lamda: maximum alpha
    :param eps: eps limit for each class
    :param delta:
    :return:
    """
    #generate grid search for alpha
    log_term = np.log(2 / delta)
    a = np.maximum(1 / eps * (1 + log_term), 0) +2
    alpha, gamma = util.limit(a, eps)
    #alpha = 100

    num_l = 10
    d = X.shape[1] + 1
    x1 = X
    num_cls = len(np.unique(y))
    #num_cls = np.max(y[:2]) + 1
    alpha_list = np.linspace(a+0.2, alpha, num= num_l)
    utility_list = np.zeros([num_l, num_cls])
    min_eigen = np.zeros_like(utility_list)# shape of number_lambda * class
    gamma_list = np.zeros_like(utility_list)
    coeff_list = [] # shape is number of lamda * cls * d+1
    hessian_list = []
    for idx, lamda in enumerate(alpha_list):
        clf = LogisticRegression(penalty='l2',C=2/lamda, solver='sag',multi_class='ovr').fit(X,y)
        # add bias column for coef
        coeff = clf.coef_   #doesn't involve bias here, bias is self.intercept_
        bias = clf.intercept_
        bias = np.expand_dims(bias, axis = 1)
        # coeff refer to theta star in paper, should be cls * d+1
        coeff =np.concatenate((coeff, bias),axis = 1)
        coeff_list.append(coeff)
        hessian_array = []
        print('lambda={} score={}'.format(lamda, clf.score(X,y)))

        p1 = clf.predict_proba(x1) #shape n*k, k is class, remove the normalize part
        p2 = np.ones([p1.shape[0],p1.shape[1]]) - p1
        p3 = p1 * p2 # shape  of p3 is n * k
        #p3 = p3.transpose() # shape of p3 is k*n
        new_col = np.ones([len(x1), 1])  # for bias term, \belta* 1
        new_X = np.append(x1, new_col, axis=1)
        for cls in range(len(p3[0])): #enumerate all class
            diag_w = np.diag(p3[:, cls])
            hessian = -np.dot(new_X.T,diag_w)
            hessian = np.dot(hessian, new_X)
            hessian_array.append(hessian)
            cur_lamda = np.linalg.cond(hessian, p=-2)
            min_eigen[idx][cls] = cur_lamda
            gamma = util.obtain_gamma(cur_lamda + lamda, eps, delta)
            gamma_list[idx][cls] = gamma
            cur_utility = 1/gamma*np.exp(np.sqrt(np.log(1/delta)/(gamma*(cur_lamda+lamda)))) # need to check
            utility_list[idx][cls] = cur_utility
        hessian_list.append(hessian_array)
    record = []
    utility_list = utility_list.transpose() #after transpose, shape should be class* number of lamda
    gamma_list = gamma_list.transpose()
    min_eigen = min_eigen.transpose()
    theta =[]
    for idx in range(num_cls):
        max_utility = np.argmin(utility_list[idx])
        cur_record = {}
        lamda  = min_eigen[idx][max_utility]
        gamma = gamma_list[idx][max_utility]
        cur_record['lamda'] = lamda + alpha_list[max_utility]
        cur_record['gamma'] = gamma
        #cur_record['hessian'] = 1/(cur_record['lamda']*cur_record['gamma'])*np.eye()
        #cur_record['theta'] = coeff[max_utility][idx]
        theta_hat = coeff_list[max_utility][idx]
        theta_hat = coeff_list[max_utility][idx] + (1/np.sqrt(cur_record['lamda']* gamma))*np.random.standard_normal(d)
        #theta_hat = np.random.multivariate_normal(coeff[max_utility][idx], hessian_list[max_utility][idx])
        theta.append(theta_hat)
    return theta


def evaluation(X, theta, y):
    """
    evaluate logistic regressian
    :param X:
    :param theta: class * dim
    :param y:
    :return:
    """
    num_cls = len(np.unique(y))
    new_col = np.ones([len(X), 1])  # for bias term, \belta* 1
    new_X = np.append(X, new_col, axis=1) #add bias term
    theta = np.array(theta)
    prob = np.dot(new_X, theta.transpose())
    prob *= -1.
    np.exp(prob, prob)
    prob += 1
    np.reciprocal(prob, prob)
    prob /= prob.sum(axis=1).reshape((prob.shape[0], -1))
    y_hat = prob.argmax(axis=1)
    y_hat = [x + np.min(y) for x in y_hat]
    sum = 0
    for idx in range(len(y)):
        if y[idx] == y_hat[idx]:
            sum+=1
    accuracy = sum / len(y)
    print('accuracy = {}'.format(accuracy))
    return accuracy

def ignore():
    from sklearn.datasets import load_iris
    X, y = load_iris(return_X_y=True)
    X = normalize(X)

def non_private(X, y):
    clf = LogisticRegression(penalty='l2', C=2 / 1, solver='sag', multi_class='ovr', n_jobs=-1).fit(X, y)

    print('standard_accuracy={}'.format(clf.score(X, y)))
    return clf.score(X, y)

def main(X_train, y_train, X_test, y_test):
    eps_set = [0.04, 0.1, 1, 3,5, 10]
    #X_train, X_test, y_train, y_test = init(csv_path)
    delta = 1e-6
    funcs = [train_log, util.cvx_objpert]
    models = []
    ac_train = []
    ac_test = []
    for eps in eps_set:
        cur_actrain = []
        cur_actest = []
        for func in funcs:
            model = train(func, X_train, y_train, eps, delta)
            cur_actrain.append(evaluation(X_train, model, y_train))
            cur_actest.append(evaluation(X_test, model, y_test))
            models.append(model)
        cur_actrain.append(non_private(X_train, y_train))
        cur_actest.append(non_private(X_test,y_test))
        ac_train.append(cur_actrain)
        ac_test.append(cur_actest)

    ac_train = np.array(ac_train)
    ac_test = np.array(ac_test)
    print(ac_train)
    print(ac_test)
    
    

    import matplotlib
    import matplotlib.pyplot as plt

    plt.loglog(eps_set, ac_train[:,0], '--r', linewidth=2)
    # plt.loglog(alpha_list, old_general, '--k', linewidth=1)
    plt.loglog(eps_set, ac_train[:,1], '-b',
               linewidth=2)  # [acgfacct.get_rdp(i+1) for i in  range(acgfacct.m)])
    plt.loglog(eps_set, ac_train[:,2], '--k',
               linewidth=2)  # [acgfacct3.get_rdp(i + 1) for i in range(acgfacct.m)])

    plt.legend(['dp_glm', 'object puerb',
                'non-private', 'Bound for Gaussian'],loc='lower right')

    plt.savefig("hhh.pdf", bbox_inches='tight')
    plt.xlabel(r'eps')
    plt.ylabel(r'accuracy')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':

    if FLAGS.dataset == 'mnist':
        X_train, y_train, X_test, y_test = input.ld_mnist(test_only=False, train_only = False)
        X_train = normalize(X_train.reshape([-1, 784]))[:1000,:]
        y_train = y_train[:1000]
        X_test = normalize(X_test.reshape([-1, 784]))

    elif FLAGS.dataset == 'adult':
        file_Name = "adult/adult.data"
        # open the file for writing
        fileObject = open(file_Name, 'rb')
        dataset = pickle.load(fileObject)
        X_train = dataset['train_data']
        y_train = dataset['train_label']
        X_test = dataset['test_data']
        y_test = dataset['test_label']

    else:
        X_train, X_test, y_train, y_test = init(csv_path)
    main( X_train, y_train, X_test, y_test)
