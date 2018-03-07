# AI for Self Driving Car

# Importing the libraries

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable

# Neural Network Class
'''In the init function we will define the architecture
of the neural network such as the input layer, which will
be composed from 5 input neurons, because we have a 5
dimentional vector encoded in the input state (3 for the
signals, and 2 for the orientation). We will define
also the hidden layers. and the the output layer which
will provide the possible action that the car can take in each
state.
Then we have the forward function, which will activate the
neurons and we will use a rectifier as activation function because we
have a non linear problem, and the rectifier activation function (ReLu),
breaks the linearity. Mainly we are creating the forward
function to return the Q values, which are the output of the neural
network each one representing an action that can be taken at each state.
Then we may take the maximum among the output Q values to choose
the best action, or we can use the softmax method.'''

# Creating the class and inheriting the functions of the nn.module class
class NN(nn.Module):

    ''' Input_vector will contain the 5 elements,
    action is the output layer which will give
    3 possible results (left, right, straight)'''

    def __init__(self, input_vector, nb_action):
        super(NN, self).__init__()# to inherit the tools of the Module class
        self.input_vector = input_vector
        self.nb_action = nb_action
        # Full connection between the input layer and the hidden layer
        self.fc1in = nn.Linear(input_vector, 64)
        # Full connection between the first hidden layer and the second one
        self.fc2 = nn.Linear(64, 32)
        # Full connection between the second hidden layer and the third one
        self.fc3 = nn.Linear(32, 16)
        # Full connection between the third hidden layer and the fourth one
        self.fc4 = nn.Linear(16, 8)
        # Full connection between the last hidden layer and the output layer
        self.fc5out = nn.Linear(8, nb_action)

    ''' State are the input entering the neural network from a specific state
    the q_values are our output, and we will be returning them '''
    def forward(self, state):
        ''' The first full connection will input the state and give the output to
        the second fully connected network'''
        # Input layer using the ReLu activation function and the state as input
        in1 = F.relu(self.fc1in(state))
        # 1st hidden layer using ReLu and 1st layer as input
        h1 = F.relu(self.fc2(in1))
        # 2nd hidden layer using ReLu and 2nd layer as input
        h2 = F.relu(self.fc3(h1))
        # 2nd hidden layer using ReLu and 2nd layer as input
        h3 = F.sigmoid(self.fc4(h2))
        # Output layer
        q_values = self.fc5out(h3)
        return q_values

# Experience replay class
''' We are implementing experience replay, because MDPs look at series of transitions
in different states which are correlated. One timestamp is not enough for the agent
to learn well and understand long term correlations, so instead of considering only
one state, we are going to take into account more previous states. In order to do this
we will append the number of states that we need to use in the memory. After that we will
be taking batches from the memory, in order to make the next update by selecting the next
action.'''

class Replay(object):

    def __init__(self, capacity):
        # Capacity will will hold a specified number of transitions
        self.capacity = capacity
        # memory will be a list that will contain the previous transitions
        self.memory = []

    # Push function
    ''' The push function will push the new transitions in the memory, and will make sure
    that the number of transitions saved does not exceed the memory length. transition will consist of 4 elements
    the first element is " S " (the last state), the 2nd one is " S' " (the next state), the 3rd
    element is " a " (the last action), and the 4th element is " r " (the last reward).
    after that we will tell to the push function, do not save more than the memory length transitions, and
    if we have a new transition that holds a timestamp > memory, delete the 1st element'''
    def push(self, transition):
        # Appending the last transition to the memory
        self.memory.append(transition)
        # If the memory length > capacity, delete the 1st element (transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # Sample functions
    '''The sample function will be used to import random samples (batches) from the memory, to improve the learning
    " samples " variable will take a random batch of samples from our memory and shape it using the zip function.
    We need the zip function, because we need to arrange the states, actions and reward elements, that we are going
    to store in our memory. We will also use the map function to convert the samples into torch variables, then we
    need to concatenate the states, actions, and rewards separatly w.r.t the 1st dimension (dimension 0) for each
    batch contained in the samples, so we will have the state, action, and reward corresponding to the same time " t "
    in each raw.
    This way we will obtain all the batches alligned and each is a torch variable '''
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        # x will be the samples, once lambda is applied onto the samples
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

# Implementing Deep Q Learning

class Dqn():
    # Creating and initializing the variables, gamma will be used as the delay
    def __init__(self, input_vector, nb_action, gamma):
        self.gamma = gamma
        '''Initializing the reward, which will be the mean of the old transitions, which we will use to evaluate
        the performance of the agent'''
        self.reward_window = []
        self.model = NN(input_vector, nb_action)
        ''' Initializing the memory, which will contain the last 100,000 transitions (we can choose more or less),
        but has shown to be a good number'''
        self.memory = Replay(100000)
        # Initializing the Optimizer, and we will choose Adam optimizer (stochastic gradient descent)
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)
        # Initializing the variables composing the transition (last state, last reward, and last action)
        '''The last state is a vector of 5 dimensions which contains 1 state of the environment, but in pytorch it
        should be a tensor and containing 1 more dimension which should be the 1st dimension that corresponds to the
        batch, because th network accepts only vectors in a batch, so a batch of input observations'''
        self.last_state = torch.Tensor(input_vector).unsqueeze(0)
        '''Actions are going to be 0, 1, or 2 and as we have seen in the map class the action to rotation vector
        contains 0 degree for the action 0, 20 degrees for action 1 and -20 degrees for action 2, so here we will
        initialize it to 0'''
        self.last_action = 0
        # initializing the reward to 0
        self.last_reward = 0

    def select_action(self, state):
        '''We will use the softmax function to select the action, which will calculate the probability (proba) of each action
        that can be taken in a given state, since in q learning we have the (action, state) combination, and softmax
        function will choose the maximum between the q values in the output layer.
        input_state is a torch tensor that contains also the gradient, but since here we don't need the gradient, because
        it is the input state, we are going to convert the torch tensor into a torch variable (to enhance the computation)
        volatile = True is used to disregard the gradient (not compute it) and this will also save us some memory.
        We will use the temperature parameter, which is a positive number and should be optimized to get a good result from
        the neural network. The neural network uses the temperature parameter to help it be sure about the selected action,
        the NN is more sure about the action that it will select when the temperature is greater than 0, the higher the
        temperature is, the higher will be the probability of the winning Q value, in other words, the more sure the NN will
        be about the chosen action, because when multiplying a probability by a high number, the number will become even higher
        which will give the NN a better confidence about choosing the corresponding probability'''
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) # T=100
        '''The softmax generates a probability distribution for each of the Q values, then we take a random draw from this
        distribution to decide the final action'''
        action = probs.multinomial()
        return action.data[0,0]# to take the corresponding action (0, 1, or 2 --> 0, 20, -20)
    # Training the Deep Neural network
    '''In this function we will implement the forward propagation and back propagation, where we are going to get our output
    compare it to the target, calculate the error and then backpropagate it through the network and update the weights using
    the stochastic gradient descent'''
    # Current_state, next_state, reward, and action are batches that we take from the memory
    def train(self, batch_state, batch_next_state, batch_reward, batch_action):
        ''' We use gather here to take only the chosen action (best action) and not all of them, unsqueeze(1), because we
        need the dimension of the action, and then we need to convert the batch into a single vector after we get the output,
        because we use batches of tensors when we are working inside the neural network, but outside the neural network we
        need to use a vector which contains the original number of dimensions (5), so we use squeeze(1)'''
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        # Taking the maximum of the Q values of the next state
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        # Choosing huber loss as a loss function
        td_loss = F.smooth_l1_loss(outputs, target)
        # Reinitializing the optimizer at each iteration using zero_grad
        self.optimizer.zero_grad()
        # Backpropagationg the loss into the network
        td_loss.backward(retain_variables = True)
        # Updating the weights
        self.optimizer.step()

    # Update function that will update the elements of the transition after reaching a new state
    def update(self, reward, new_signal):
        #1st we want to update the new state
        '''The new state depends on the signal that the sensors has detected, so the state contains signal1, signal2, signal3
        orientation and -orientation that we already defined in the map class, and we need to convert this vector into a torch
        tensor as before'''
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        # We should update the memory, because now we have the new state so we need to append it to the memory
        # We need the LongTensor object that holds an integer, because the output of the action is 0, 1, or 2
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        # Playing the new action after reaching the new state
        action = self.select_action(new_state)
        ''' Learning after reaching more than 100 transitions in the memory, so the agent will be able to learn from
        the past transitions, we will take a sample of 100 transition each time'''
        if len(self.memory.memory) > 100:
            '''Take a random sample of 100 transitions that we can get with the sample function that returns the batches:
            the state at time t, the state at time t+1, the action at time t, and the reward at time t, so we need new
            variables which are going to be the batch at time t, batch at t+1, action at t, and reward at t'''
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            # Learning with 100 current states, 100 next states, 100 rewards, and 100 actions
            self.train(batch_state, batch_next_state, batch_reward, batch_action)
        # Update the played action (last action), the last reward, and the last state which became the new state.
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        # Updating the reward window
        '''Which will take the mean of the last 100 rewards and keep track of the training.
        The reward window is of fixed size, shifting with time to show the evolution of the reward'''
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        # Returning the action
        return action

    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1.)

    def save(self):
        torch.save({'state_dict': self.model.state_dict(),
                    'optimizer' : self.optimizer.state_dict(),
                   }, 'last_brain.pth')

    def load(self):
        if os.path.isfile('last_brain.pth'):
            print("=> loading checkpoint... ")
            checkpoint = torch.load('last_brain.pth')
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            print("done !")
        else:
            print("no checkpoint found...")
