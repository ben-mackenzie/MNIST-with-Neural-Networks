# Do not use packages that are not in standard distribution of python
import numpy as np

class _baseNetwork:
    def __init__(self, input_size=28 * 28, num_classes=10):

        self.input_size = input_size
        self.num_classes = num_classes

        self.weights = dict()
        self.gradients = dict()

    def _weight_init(self):
        pass

    def forward(self):
        pass

    def softmax(self, scores):
        '''
        Compute softmax scores given the raw output from the model

        :param scores: raw scores from the model (N, num_classes)
        :return:
            prob: softmax probabilities (N, num_classes)
        '''
        prob = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Calculate softmax scores of input images                            #
        #############################################################################
        # https://deepnotes.io/softmax-crossentropy
        prob = []
        for score in scores:
            exps = np.exp(score - np.max(score)) # changed my softmax to stable version from DeepNotes
            p = exps / np.sum(exps)
            prob.append(p)
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################

        return prob

    def cross_entropy_loss(self, x_pred, y):
        '''
        Compute Cross-Entropy Loss based on prediction of the network and labels
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The computed Cross-Entropy Loss
        '''
        loss = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement Cross-Entropy Loss                                        #
        #############################################################################
        # https://deepnotes.io/softmax-crossentropy
        m = len(y)
        #p = np.max(x_pred, axis = 1)
        #log_likelihood = -np.log(p) #got the numpy implementation from the article above
        r = np.arange(0,m,1)
        x_pred = np.asarray(x_pred)
        log_likelihood = -np.log(x_pred[r,y])
        loss = np.sum(log_likelihood) / m
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return loss

    def compute_accuracy(self, x_pred, y):
        '''
        Compute the accuracy of current batch
        :param x_pred: Raw prediction from the two-layer net (N, num_classes)
        :param y: Labels of instances in the batch
        :return: The accuracy of the batch
        '''
        acc = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the accuracy function                                     #
        #############################################################################
        x_preds = []
        for prediction in x_pred:
            xp = np.argmax(prediction)
            x_preds.append(xp)
        correct = len(np.where(x_preds==y))
        total_examples = len(y)
        correct = np.sum(x_preds == y)
        acc = correct/total_examples
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return acc

    def sigmoid(self, X):
        '''
        Compute the sigmoid activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the sigmoid activation is applied to the input (N, num_classes)
        '''
        out = None
        #############################################################################
        # TODO: Comput the sigmoid activation on the input                          #
        #############################################################################
        '''
        activated_data = []
        for row in X:
            activated_row = 1/(1 + np.exp(-row))
            activated_data.append(activated_row)
        out = activated_data
        '''
        X = np.asarray(X)
        out = 1/(1 + np.exp(-X))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def sigmoid_dev(self, x):
        '''
        The analytical derivative of sigmoid function at x
        :param x: Input data
        :return: The derivative of sigmoid function at x
        '''
        ds = None
        #############################################################################
        # TODO:                                                                     #
        #    1) Implement the derivative of Sigmoid function                        #
        #############################################################################
        sigmoid_x = self.sigmoid(x)
        ds = sigmoid_x*(np.subtract(1.0,sigmoid_x))
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return ds

    def ReLU(self, X):
        '''
        Compute the ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: the value after the ReLU activation is applied to the input (N, num_classes)
        '''
        out = None
        #############################################################################
        # TODO: Comput the ReLU activation on the input                          #
        #############################################################################
        relu_rows = []
        for row in X:
            relu = np.maximum(0,row)
            relu_rows.append(relu)
        out = relu_rows
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out

    def ReLU_dev(self,X):
        '''
        Compute the gradient ReLU activation for the input

        :param X: the input data coming out from one layer of the model (N, num_classes)
        :return:
            out: gradient of ReLU given input X
        '''
        out = None
        #############################################################################
        # TODO: Compute the gradient of ReLU activation                              #
        #############################################################################
        #https://stackoverflow.com/questions/46411180/implement-relu-derivative-in-python-numpy
        out = (X > 0) * 1 #<< was able to condense my conditional statements using this!
        #############################################################################
        #                              END OF YOUR CODE                             #
        #############################################################################
        return out
