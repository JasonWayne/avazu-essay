from datetime import datetime
from datetime import datetime
from csv import DictReader
from math import exp, log, sqrt
import sys


# TL; DR, the main training process starts on line: 282,
# you may want to start reading the code from there


##############################################################################
# parameters #################################################################
##############################################################################

# train cmd
# python base/ftrl/ftrl.py base/tr.r0.app.sp base/va.r0.app.sp submission_app.csv y

# A, paths
# train = 'base/tr.r0.app.sp'
# test = 'base/va.r0.app.sp'                 # path to testing file
arg_count = len(sys.argv)
if arg_count < 3:
    print "not enough args, quit"
    exit(1)
train = sys.argv[1]
test = sys.argv[2]
submission = sys.argv[3]  # path of to be outputted submission file
interaction = sys.argv[4] if arg_count > 4 else 'n'
interaction_file_path = sys.argv[5] if arg_count > 5 else ''
print "train -> {0}, test -> {1}, submission -> {2}".format(train, test, submission)

# B, model
alpha = .1  # learning rate
beta = 1.   # smoothing parameter for adaptive learning rate
L1 = 1.     # L1 regularization, larger value means more regularized
L2 = 1.     # L2 regularization, larger value means more regularized

# C, feature/hash trick
# D = 2 ** 20              # number of weights to use
D = 1000000
do_interactions = True if interaction == 'y' else False # whether to enable poly2 feature interactions

# D, training/validation
epoch = 3      # learn training data for N passes
holdout = 100  # use every N training instance for holdout validation



##############################################################################
# class, function, generator definitions #####################################
##############################################################################

# each class below is a learning algorithm

class logistic_regression(object):
    ''' Classical logistic regression

        This class (algorithm) is not used in this code, it is putted here
        for a quick reference in hope to make the following (more complex)
        algorithm more understandable.
    '''

    def __init__(self, alpha, D, interaction=False):
        # parameters
        self.alpha = alpha

        # model
        self.w = [0.] * D

    def predict(self, x):
        # parameters
        alpha = self.alpha

        # model
        w = self.w

        # wTx is the inner product of w and x
        wTx = sum(w[i] for i in x)

        # bounded sigmoid function, this is the probability of being clicked
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        # parameter
        alpha = self.alpha

        # model
        w = self.w

        # gradient under logloss
        g = p - y

        # update w
        for i in x:
            w[i] += g * alpha


class ftrl_proximal(object):
    ''' Our main algorithm: Follow the regularized leader - proximal

        In short,
        this is an adaptive-learning-rate sparse logistic-regression with
        efficient L1-L2-regularization

        Reference:
        http://www.eecs.tufts.edu/~dsculley/papers/ad-click-prediction.pdf
    '''

    def __init__(self, alpha, beta, L1, L2, D, interaction=False):
        # parameters
        self.alpha = alpha
        self.beta = beta
        self.L1 = L1
        self.L2 = L2

        # feature related parameters
        self.D = D
        self.interaction = interaction
        self.interaction_dict = {}

        # model
        # n: squared sum of past gradients
        # z: weights
        # w: lazy weights
        self.n = [0.] * D
        self.z = [0.] * D
        self.w = [0.] * D  # use this for execution speed up
        # self.w = {}  # use this for memory usage reduction

    def _indices(self, x):
        ''' A helper generator that yields the indices in x

            The purpose of this generator is to make the following
            code a bit cleaner when doing feature interaction.
        '''

        for i in x:
            yield i

        if self.interaction:
            D = self.D
            L = len(x)
            for i in xrange(1, L):  # skip bias term, so we start at 1
                for j in xrange(i+1, L):
                    yield (i * j) % D

    def predict(self, x):
        ''' Get probability estimation on x

            INPUT:
                x: features

            OUTPUT:
                probability of p(y = 1 | x; w)
        '''

        # parameters
        alpha = self.alpha
        beta = self.beta
        L1 = self.L1
        L2 = self.L2

        # model
        n = self.n
        z = self.z
        w = self.w  # use this for execution speed up
        # w = {}  # use this for memory usage reduction

        # wTx is the inner product of w and x
        wTx = 0.
        for i in self._indices(x):
            sign = -1. if z[i] < 0 else 1.  # get sign of z[i]

            # build w on the fly using z and n, hence the name - lazy weights -
            if sign * z[i] <= L1:
                # w[i] vanishes due to L1 regularization
                w[i] = 0.
            else:
                # apply prediction time L1, L2 regularization to z and get w
                w[i] = (sign * L1 - z[i]) / ((beta + sqrt(n[i])) / alpha + L2)

            wTx += w[i]

        self.w = w

        # bounded sigmoid function, this is the probability estimation
        return 1. / (1. + exp(-max(min(wTx, 35.), -35.)))

    def update(self, x, p, y):
        ''' Update model using x, p, y

            INPUT:
                x: feature, a list of indices
                p: click probability prediction of our model
                y: answer

            MODIFIES:
                self.n: increase by squared gradient
                self.z: weights
        '''

        # parameter
        alpha = self.alpha

        # model
        n = self.n
        z = self.z
        w = self.w  # no need to change this, it won't gain anything

        # gradient under logloss
        g = p - y

        # update z and n
        for i in self._indices(x):
            sigma = (sqrt(n[i] + g * g) - sqrt(n[i])) / alpha
            z[i] += g - sigma * w[i]
            n[i] += g * g


def logloss(p, y):
    ''' FUNCTION: Bounded logloss

        INPUT:
            p: our prediction
            y: real answer

        OUTPUT:
            logarithmic loss of p given y
    '''

    p = max(min(p, 1. - 10e-15), 10e-15)
    return -log(p) if y == 1. else -log(1. - p)


def data(path, D):
    ''' GENERATOR: Apply hash-trick to the original csv row
                   and for simplicity, we one-hot-encode everything

        INPUT:
            path: path to training or testing file
            D: the max index that we can hash to

        YIELDS:
            ID: id of the instance, mainly useless
            x: a list of hashed and one-hot-encoded 'indices'
               we only need the index since all values are either 0 or 1
            y: y = 1 if we have a click, else we have y = 0
    '''

    row_length = -1
    for t, line in enumerate(open(path)):
        # drop first line
        if t == 0:
            continue
        line = line.strip().split(" ")
        if row_length == -1:
            row_length = len(line)
        # process id
        ID = line[0]

        # process clicks
        y = float(line[1])

        # build x
        x = [0]  # 0 is the index of the bias term
        # for key in xrange(2, row_length):  # sort is for preserving feature ordering
        for key in xrange(3, 4):  # sort is for preserving feature ordering
            value = line[key]

            # one-hot encode everything with hash trick
            index = abs(hash(str(key) + '_' + value)) % D
            x.append(index)

        yield t, ID, x, y


##############################################################################
# start training #############################################################
##############################################################################

start = datetime.now()

# initialize ourselves a learner
learner = ftrl_proximal(alpha, beta, L1, L2, D, interaction=do_interactions)

# start training
for e in xrange(epoch):
    loss = 0.
    count = 0

    for t, ID, x, y in data(train, D):  # data is a generator
        #  t: just a instance counter
        # ID: id provided in original data
        #  x: features
        #  y: label (click)

        # step 1, get prediction from learner
        p = learner.predict(x)

        if t % holdout == 0:
            # step 2-1, calculate holdout validation loss
            #           we do not train with the holdout data so that our
            #           validation loss is an accurate estimation of
            #           the out-of-sample error
            loss += logloss(p, y)
            count += 1
        else:
            # step 2-2, update learner with label (click) information
            learner.update(x, p, y)

        if t % 2500000 == 0 and t > 1:
            print(' %s\tencountered: %d\tcurrent logloss: %f' % (
                datetime.now(), t, loss/count))

    print('Epoch %d finished, holdout logloss: %f, elapsed time: %s' % (
        e, loss/count, str(datetime.now() - start)))


##############################################################################
# start testing, and build Kaggle's submission file ##########################
##############################################################################

with open(submission, 'w') as outfile:
    for t, ID, x, y in data(test, D):
        p = learner.predict(x)
        outfile.write('%s,%s\n' % (ID, str(p)))
