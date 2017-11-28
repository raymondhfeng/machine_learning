import numpy as np

import backend
import nn

class Model(object):
    """Base model class for the different applications"""
    def __init__(self):
        self.get_data_and_monitor = None
        self.learning_rate = 0.0

    def run(self, x, y=None):
        raise NotImplementedError("Model.run must be overriden by subclasses")

    def train(self):
        """
        Train the model.

        get_data_and_monitor will yield data points one at a time. In between
        yielding data points, it will also monitor performance, draw graphics,
        and assist with automated grading. The model (self) is passed as an
        argument to get_data_and_monitor, which allows the monitoring code to
        evaluate the model on examples from the validation set.
        """
        for x, y in self.get_data_and_monitor(self):
            graph = self.run(x, y)
            graph.backprop()
            graph.step(self.learning_rate)

class RegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers. The network should be sufficiently large to be able
    to approximate sin(x) on the interval [-2pi, 2pi] to reasonable precision.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = 0.5
        self.graph = None

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs y are known during training, but not at test time.
        If correct outputs y are provided, this method must construct and
        return a nn.Graph for computing the training loss. If y is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "* YOUR CODE HERE *"
        if y is not None:
            # At training time, the correct output y is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            n = 5
            if not self.graph:
                self.l = []
                for i in range(0, n):
                    self.l.append(nn.Variable(len(x), len(x)))
                for i in range(0, n):
                    self.l.append(nn.Variable(len(x), 1))
                self.graph = nn.Graph(self.l)
                input_x = nn.Input(self.graph,x)
                input_y = nn.Input(self.graph,y)
                mult = nn.MatrixMultiply(self.graph, self.l[0], input_x)
                add = nn.MatrixVectorAdd(self.graph, mult, self.l[n])
                for i in range(0, n):
                    relu = nn.ReLU(self.graph, add)
                    mult = nn.MatrixMultiply(self.graph, self.l[i], relu)
                    add = nn.MatrixVectorAdd(self.graph, self.l[n + i], mult)
                loss = nn.SquareLoss(self.graph, add, input_y)
                return self.graph
            else:
                self.graph = nn.Graph(self.l)
                input_x = nn.Input(self.graph,x)
                input_y = nn.Input(self.graph,y)
                mult = nn.MatrixMultiply(self.graph, self.l[0], input_x)
                add = nn.MatrixVectorAdd(self.graph, mult, self.l[n])
                for i in range(0, n):
                    relu = nn.ReLU(self.graph, add)
                    mult = nn.MatrixMultiply(self.graph, self.l[i], relu)
                    add = nn.MatrixVectorAdd(self.graph, self.l[n + i], mult)
                loss = nn.SquareLoss(self.graph, add, input_y)
                return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array

#             top_vec = self.graph.get_output(self.graph.get_nodes()[-4])
#             bot_vec = self.graph.get_output(self.graph.get_nodes()[-5])
#             # print(top_vec,bot_vec)
#             return np.concatenate((top_vec, bot_vec), axis=0)

            # top_add = self.graph.get_output(self.graph.get_nodes()[-4])
            # bot_add = self.graph.get_output(self.graph.get_nodes()[-5])
            # return (top_add + bot_add) * (0.5)

            return self.graph.get_output(self.graph.get_nodes()[-2])
              

class OddRegressionModel(Model):
    """
    A neural network model for approximating a function that maps from real
    numbers to real numbers.

    Unlike RegressionModel, the OddRegressionModel must be structurally
    constrained to represent an odd function, i.e. it must always satisfy the
    property f(x) = -f(-x) at all points during training.
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_regression
        
        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = 0.1
        self.graph = None
        self.l = []

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct outputs y are known during training, but not at test time.
        If correct outputs y are provided, this method must construct and
        return a nn.Graph for computing the training loss. If y is None, this
        method must instead return predicted y-values.

        Inputs:
            x: a (batch_size x 1) numpy array
            y: a (batch_size x 1) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 1) numpy array of predicted y-values

        Note: DO NOT call backprop() or step() inside this method!
        """
        "* YOUR CODE HERE *"
        n = 4
        if not self.graph:
                w1 = nn.Variable(1, 50)          
                w2 = nn.Variable(50, 50)
                w3 = nn.Variable(50, 1)
                b1 = nn.Variable(1, 50)
                b2 = nn.Variable(1, 50)
                b3 = nn.Variable(1, 1)
                self.l = [w1,w2,w3,b1,b2,b3]
                self.graph = nn.Graph(self.l)
        self.graph = nn.Graph(self.l)
        input_x = nn.Input(self.graph,x)
        if y is not None: #<--- THIS LITTLE CONDITIONAL SO IMPORTANT HFS
            input_y = nn.Input(self.graph,y)
        input_neg = nn.Input(self.graph, np.matrix([-1.])) #Tx1
        mult = nn.MatrixMultiply(self.graph, input_x, self.l[0]) #Tx50
        add = nn.MatrixVectorAdd(self.graph, mult, self.l[3]) 
        relu = nn.ReLU(self.graph, add)
        mult2 = nn.MatrixMultiply(self.graph, relu, self.l[1]) #Tx50
        add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[4]) #Tx50
        relu2 = nn.ReLU(self.graph, add2)
        mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[2]) #Tx1
        add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[5])
        ad = add3

        neg = nn.MatrixMultiply(self.graph, input_x, input_neg)
        mult = nn.MatrixMultiply(self.graph, neg, self.l[0])
        add = nn.MatrixVectorAdd(self.graph, mult, self.l[3])
        relu = nn.ReLU(self.graph, add)
        mult2 = nn.MatrixMultiply(self.graph, relu, self.l[1])
        add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[4])
        relu2 = nn.ReLU(self.graph, add2)
        mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[2])
        add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[5])
        sb = nn.MatrixMultiply(self.graph, add3, input_neg) #-g(-x)
        sub = nn.MatrixVectorAdd(self.graph, ad, sb) #g(x) - g(-x)
            
        if y is not None:
            # At training time, the correct output y is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            loss = nn.SquareLoss(self.graph, sub, input_y)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            #print(self.graph.get_output(self.graph.get_nodes()[-1]))
            return self.graph.get_output(self.graph.get_nodes()[-1])
            
              

class DigitClassificationModel(Model):
    """
    A model for handwritten digit classification using the MNIST dataset.

    Each handwritten digit is a 28x28 pixel grayscale image, which is flattened
    into a 784-dimensional vector for the purposes of this model. Each entry in
    the vector is a floating point number between 0 and 1.

    The goal is to sort each digit into one of 10 classes (number 0 through 9).

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_digit_classification

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        "*** YOUR CODE HERE ***"
        self.learning_rate = 0.2
        self.graph = None

    def run(self, x, y=None):
        """
        Runs the model for a batch of examples.

        The correct labels are known during training, but not at test time.
        When correct labels are available, `y` is a (batch_size x 10) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should predict a (batch_size x 10) numpy array of scores,
        where higher scores correspond to greater probability of the image
        belonging to a particular class. You should use `nn.SoftmaxLoss` as your
        training loss.

        Inputs:
            x: a (batch_size x 784) numpy array
            y: a (batch_size x 10) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 10) numpy array of scores (aka logits)
        """
        "*** YOUR CODE HERE ***"
        if len(x) == 1:
            return 0
        if not self.graph:
                w1 = nn.Variable(784, 500)          
                w2 = nn.Variable(500, 500)       
                w3 = nn.Variable(500, 10)       
                b1 = nn.Variable(1, 500)
                b2 = nn.Variable(1, 500)
                b3 = nn.Variable(1, 10)
                self.l = [w1,w2,w3,b1,b2,b3]
                self.graph = nn.Graph(self.l)
        self.graph = nn.Graph(self.l)
        input_x = nn.Input(self.graph,x) #Tx784
        if y is not None: #<--- THIS LITTLE CONDITIONAL SO IMPORTANT HFS
            input_y = nn.Input(self.graph,y)
        mult = nn.MatrixMultiply(self.graph, input_x, self.l[0]) #Tx50
        add = nn.MatrixVectorAdd(self.graph, mult, self.l[3]) 
        relu = nn.ReLU(self.graph, add)
        mult2 = nn.MatrixMultiply(self.graph, relu, self.l[1]) #Tx50
        add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[4]) #Tx50
        relu2 = nn.ReLU(self.graph, add2)
        mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[2]) 
        add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[5])
        if y is not None:
            # At training time, the correct output `y` is known.
            # Here, you should construct a loss node, and return the nn.Graph
            # that the node belongs to. The loss node must be the last node
            # added to the graph.
            loss = nn.SoftmaxLoss(self.graph, add3, input_y)
            return self.graph
        else:
            # At test time, the correct output is unknown.
            # You should instead return your model's prediction as a numpy array
            #print(self.graph.get_output(self.graph.get_nodes()[-1]))
            return self.graph.get_output(self.graph.get_nodes()[-1])


class DeepQModel(Model):
    """
    A model that uses a Deep Q-value Network (DQN) to approximate Q(s,a) as part
    of reinforcement learning.

    (We recommend that you implement the RegressionModel before working on this
    part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_rl

        self.num_actions = 2
        self.state_size = 4

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = 0.01
        self.graph = None

    def run(self, states, Q_target=None):
        """
        Runs the DQN for a batch of states.

        The DQN takes the state and computes Q-values for all possible actions
        that can be taken. That is, if there are two actions, the network takes
        as input the state s and computes the vector [Q(s, a_1), Q(s, a_2)]

        When Q_target == None, return the matrix of Q-values currently computed
        by the network for the input states.

        When Q_target is passed, it will contain the Q-values which the network
        should be producing for the current states. You must return a nn.Graph
        which computes the training loss between your current Q-value
        predictions and these target values, using nn.SquareLoss.

        Inputs:
            states: a (batch_size x 4) numpy array
            Q_target: a (batch_size x 2) numpy array, or None
        Output:
            (if Q_target is not None) A nn.Graph instance, where the last added
                node is the loss
            (if Q_target is None) A (batch_size x 2) numpy array of Q-value
                scores, for the two actions
        """
        "* YOUR CODE HERE *"
        if not self.graph:
            w1 = nn.Variable(4, 50)          
            w2 = nn.Variable(50, 50)
            w3 = nn.Variable(50, 2)
            b1 = nn.Variable(1, 50)
            b2 = nn.Variable(1, 50)
            b3 = nn.Variable(1, 2)
            self.l = [w1,w2,w3,b1,b2,b3]
            self.graph = nn.Graph(self.l)
        self.graph = nn.Graph(self.l)
        input_x = nn.Input(self.graph,states)
        if Q_target is not None: #<--- THIS LITTLE CONDITIONAL SO IMPORTANT HFS
            input_y = nn.Input(self.graph, Q_target) #Tx4
        mult = nn.MatrixMultiply(self.graph, input_x, self.l[0]) #Tx4 
        add = nn.MatrixVectorAdd(self.graph, mult, self.l[3]) 
        relu = nn.ReLU(self.graph, add)
        mult2 = nn.MatrixMultiply(self.graph, relu, self.l[1]) #Tx50
        add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[4]) #Tx50
        relu2 = nn.ReLU(self.graph, add2)
        mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[2]) #Tx1
        add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[5])
        if Q_target is not None:
            "* YOUR CODE HERE *"
            loss = nn.SquareLoss(self.graph, add3, input_y)
            return self.graph
        else:
            "* YOUR CODE HERE *"
            return self.graph.get_output(self.graph.get_nodes()[-1])

    def get_action(self, state, eps):
        """
        Select an action for a single state using epsilon-greedy.

        Inputs:
            state: a (1 x 4) numpy array
            eps: a float, epsilon to use in epsilon greedy
        Output:
            the index of the action to take (either 0 or 1, for 2 actions)
        """
        if np.random.rand() < eps:
            return np.random.choice(self.num_actions)
        else:
            scores = self.run(state)
            return int(np.argmax(scores))


class LanguageIDModel(Model):
    """
    A model for language identification at a single-word granularity.

    (See RegressionModel for more information about the APIs of different
    methods here. We recommend that you implement the RegressionModel before
    working on this part of the project.)
    """
    def __init__(self):
        Model.__init__(self)
        self.get_data_and_monitor = backend.get_data_and_monitor_lang_id

        # Our dataset contains words from five different languages, and the
        # combined alphabets of the five languages contain a total of 47 unique
        # characters.
        # You can refer to self.num_chars or len(self.languages) in your code
        self.num_chars = 47
        self.languages = ["English", "Spanish", "Finnish", "Dutch", "Polish"]

        # Remember to set self.learning_rate!
        # You may use any learning rate that works well for your architecture
        "* YOUR CODE HERE *"
        self.learning_rate = 0.03
        self.graph = None

    def run(self, xs, y=None):
        """
        Runs the model for a batch of examples.

        Although words have different lengths, our data processing guarantees
        that within a single batch, all words will be of the same length (L).

        Here xs will be a list of length L. Each element of xs will be a
        (batch_size x self.num_chars) numpy array, where every row in the array
        is a one-hot vector encoding of a character. For example, if we have a
        batch of 8 three-letter words where the last word is "cat", we will have
        xs[1][7,0] == 1. Here the index 0 reflects the fact that the letter "a"
        is the inital (0th) letter of our combined alphabet for this task.

        The correct labels are known during training, but not at test time.
        When correct labels are available, y is a (batch_size x 5) numpy
        array. Each row in the array is a one-hot vector encoding the correct
        class.

        Your model should use a Recurrent Neural Network to summarize the list
        xs into a single node that represents a (batch_size x hidden_size)
        array, for your choice of hidden_size. It should then calculate a
        (batch_size x 5) numpy array of scores, where higher scores correspond
        to greater probability of the word originating from a particular
        language. You should use nn.SoftmaxLoss as your training loss.

        Inputs:
            xs: a list with L elements (one per character), where each element
                is a (batch_size x self.num_chars) numpy array
            y: a (batch_size x 5) numpy array, or None
        Output:
            (if y is not None) A nn.Graph instance, where the last added node is
                the loss
            (if y is None) A (batch_size x 5) numpy array of scores (aka logits)

        Hint: you may use the batch_size variable in your code
        """
        batch_size = xs[0].shape[0]
        if not self.graph:
            w1 = nn.Variable(47, 47) #         
            w2 = nn.Variable(47, 47)
            w3 = nn.Variable(50, 2)
            b1 = nn.Variable(1, 47) #
            b2 = nn.Variable(1, 47) 
            # b3 = nn.Variable(1, 2) 
            h0 = nn.Variable(1, 47)

            w3 = nn.Variable(47, 47)  
            w4 = nn.Variable(47, 100) # 
            w6 = nn.Variable(100, 5) #
            b3 = nn.Variable(1, 47) 
            b4 = nn.Variable(1, 100) #
            b6 = nn.Variable(1, 5) # 

            w5 = nn.Variable(47, 47)
            b5 = nn.Variable(1, 47) 
            
            self.l = [w1,w2,b1,b2,h0,w3,w4,b3,b4, w5, b5,w6,b6]
            #         #     #            #    #           # #
        self.graph = nn.Graph(self.l)
        
        
        
 

        "* YOUR CODE HERE *"
        char_inputs = [] 
        h = self.l[4]
        zero = np.zeros((batch_size, 47))
        zeroInput = nn.Input(self.graph,zero)
        z = nn.MatrixVectorAdd(self.graph, zeroInput, h)
        h = z
        for i in range(len(xs)):
            char_inputs.append(nn.Input(self.graph, xs[i])) 
            incorporate = nn.MatrixVectorAdd(self.graph, h, char_inputs[i]) #Tx47 x 
            mult = nn.MatrixMultiply(self.graph, incorporate, self.l[0]) #Tx47
            add = nn.MatrixVectorAdd(self.graph, mult, self.l[2]) 
            relu = nn.ReLU(self.graph, add)
            # mult2 = nn.MatrixMultiply(self.graph, relu, self.l[1]) #Tx47
            # add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[3]) #Tx47
            # relu2 = nn.ReLU(self.graph, add2)
            # mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[9]) #Tx47
            # add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[10]) #Tx47
            # relu3 = nn.ReLU(self.graph, add3)
            h = relu
        # mult = nn.MatrixMultiply(self.graph, h, self.l[5]) #Tx47
        # add = nn.MatrixVectorAdd(self.graph, mult, self.l[7]) #Tx47
        # relu = nn.ReLU(self.graph, add)
        mult2 = nn.MatrixMultiply(self.graph, h, self.l[6]) #Tx5
        add2 = nn.MatrixVectorAdd(self.graph, mult2, self.l[8]) #Tx5
        relu2 = nn.ReLU(self.graph, add2)
        mult3 = nn.MatrixMultiply(self.graph, relu2, self.l[11]) #Tx5
        add3 = nn.MatrixVectorAdd(self.graph, mult3, self.l[12]) #Tx5
        if y is not None:
            "* YOUR CODE HERE *"
 
            input_y = nn.Input(self.graph, y) #Tx5
            #print(self.graph.get_output(input_y))
            loss = nn.SoftmaxLoss(self.graph, add3, input_y)
            return self.graph
        else:
            "* YOUR CODE HERE *"
            return self.graph.get_output(self.graph.get_nodes()[-1])
