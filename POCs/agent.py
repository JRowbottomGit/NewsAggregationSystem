############################################################################
############################################################################
# THIS IS THE ONLY FILE YOU SHOULD EDIT
#
#
# Agent must always have these five functions:
#     __init__(self)
#     has_finished_episode(self)
#     get_next_action(self, state)
#     set_next_state_and_distance(self, next_state, distance_to_goal)
#     get_greedy_action(self, state)
#
#
# You may add any other functions as you wish
############################################################################
############################################################################
import numpy as np
import torch
import collections

class Agent:
    # Function to initialise the agent
    def __init__(self):
        # hyper paramters
        self.episode_length = 500
        self.epsilon = 0.9 #1
        self.epsilon_min = 0.2
        self.delta = 0.0001 #0.02
        self.epsilon_method = 'Q_epsilon_greedy' #['epsilon_greedy','Q_epsilon_greedy','Q_then_epsilon_greedy']
        self.replay_method = 'prioritised_experience' # ['random','prioritised_experience']
        self.target_network_reset = 100
        #instance variables
        self.total_reward = 0.0
        self.num_steps_taken = 0 # Reset the total number of steps which the agent has taken
        self.state = None # The state variable stores the latest state of the agent in the environment
        self.action = None # The action variable stores the latest action which the agent has applied to the environment
        #self.reset()# Reset the agent.
        self.distance = 0.0 #distance to goal from last move
        self.dqn = DQN(replay_method = self.replay_method) # Create a DQN (Deep Q-Network)

    # Function to check whether the agent has reached the end of an episode
    def has_finished_episode(self):
        if self.num_steps_taken % self.episode_length == 0:
            self.total_reward = 0.0 #needs to reset reward here
            return True
        else:
            return False

    # Function to convert discrete action (as used by a DQN) to a continuous action (as used by the environment).
    def _discrete_action_to_continuous(self, discrete_action):

        if self.distance  < 0.1:
            size = 1
        else:
            size = 2
        if discrete_action == 0:  # Move right
            continuous_action = size * np.array([0.01, 0], dtype=np.float32)
        elif discrete_action == 1:  # Move left
            continuous_action = size * np.array([-0.01, 0], dtype=np.float32)
        elif discrete_action == 2:  # Move up
            continuous_action = size * np.array([0, 0.01], dtype=np.float32)
        elif discrete_action == 3:  # Move down
            continuous_action = size * np.array([0, -0.01], dtype=np.float32)

        return continuous_action

    # Function to get the next action, using whatever method you like
    def get_next_action(self, state):
        # Here, the action is random, but you can change this
        #action = np.random.uniform(low=-0.01, high=0.01, size=2).astype(np.float32)
        # Update the number of steps which the agent has taken
        self.num_steps_taken += 1
        # Store the state; this will be used later, when storing the transition
        self.state = state
                ###can't use this hueristic to see if stuck as not directly from distance or q network
                # if self.action == 4:
                #     epsilon_discrete_action = self.epsilon_greedy_step(state, 0.75)  ##note epsilon value of 0.75 is purely random
                # else:

        #ask epsilon greed for next intended action:
        #can't compose transition yet as need to wait for feedback from train_test
        epsilon_discrete_action = self.epsilon_greedy_step(state, self.epsilon)
        # Convert the discrete action into a continuous action.
        continuous_action = self._discrete_action_to_continuous(int(epsilon_discrete_action))
        return continuous_action

    def dqn_training_manager(self,mini_batch_transition):
        # ###this stuff to train the NN
        if self.num_steps_taken <= self.dqn.batch_size:
        #epsilon = max(epsilon - delta, self.epsilon_min) #don't update epsilon for steps while buiding batch
            pass #already taken the step and appended transition to buffer in get_next action
        else:
            #self.epsilon = max(self.epsilon - self.delta, self.epsilon_min)
            self.epsilon = max(self.epsilon - self.delta, self.epsilon_min, self.distance*0.8)

            loss = self.dqn.train_q_network(mini_batch_transition) #do a batch training on the DQN
        ## update target
        if self.num_steps_taken % self.target_network_reset == 0: #target network stability
            self.dqn.target_update()
        #return loss

    def epsilon_shuffler(self,discrete_action,epsilon):
        if discrete_action == 0:
            shuffled_action = np.random.choice([0,1,2,3], p=[1-epsilon, epsilon/3, epsilon/3, epsilon/3])
        elif discrete_action == 1:
            shuffled_action = np.random.choice([0,1,2,3], p=[epsilon/3, 1-epsilon, epsilon/3, epsilon/3])
        elif discrete_action == 2:
            shuffled_action = np.random.choice([0,1,2,3], p=[epsilon/3, epsilon/3, 1-epsilon, epsilon/3])
        elif discrete_action == 3:
            shuffled_action = np.random.choice([0,1,2,3], p=[epsilon/3, epsilon/3, epsilon/3, 1-epsilon])

        return shuffled_action

    def Q_weighted_shuffler(self,network_prediction):
        q_value, discrete_action_tensor = network_prediction.max(0)
        discrete_action = discrete_action_tensor.item()
        Qt = network_prediction.sum().item()
        network_prediction_np = network_prediction.detach().numpy()
        network_prediction_argsort = np.argsort(network_prediction_np, axis=0) #list of indices of ascending actions
        network_prediction_double_argsort = np.argsort(network_prediction_argsort, axis=0) #list of rankings for each

        p0 = network_prediction_np[0]/Qt
        p1 = network_prediction_np[1]/Qt
        p2 = network_prediction_np[2]/Qt
        p3 = network_prediction_np[3]/Qt

        rand = np.random.uniform(0,1)
        index = 4
        if rand <= p0:
            index = 0
        elif p0 < rand and rand <= p0 + p1:
            index = 1
        elif p0 + p1 <rand and rand <= p0 + p1 + p2:
            index = 2
        elif p0 + p1 +p2 < rand and rand <= 1:
            index = 3
        return index

    def Q_epsilon_shuffler(self,network_prediction,epsilon):

        q_value, discrete_action_tensor = network_prediction.max(0)
        discrete_action = discrete_action_tensor.item()
        Qt = network_prediction.sum().item()
        network_prediction_np = network_prediction.detach().numpy()
        network_prediction_argsort = np.argsort(network_prediction_np, axis=0) #list of indices of ascending actions
        network_prediction_double_argsort = np.argsort(network_prediction_argsort, axis=0) #list of rankings for each

        index_pmax = network_prediction_argsort[3]
        index_phighermid = network_prediction_argsort[2]
        index_plowermid = network_prediction_argsort[1]
        index_pmin = network_prediction_argsort[0]
        pmax = 1 - epsilon * (Qt - network_prediction_np[index_pmax])/Qt
        phighermid = epsilon * (Qt - network_prediction_np[index_pmax])/Qt *network_prediction_np[index_phighermid]/(Qt-network_prediction_np[index_pmax])
        plowermid = epsilon * (Qt - network_prediction_np[index_pmax])/Qt *network_prediction_np[index_plowermid]/(Qt-network_prediction_np[index_pmax])
        pmin = epsilon * (Qt - network_prediction_np[index_pmax])/Qt *network_prediction_np[index_pmin]/(Qt-network_prediction_np[index_pmax])
        #print('1?')
        #print(pmax + phighermid + plowermid + pmin)
        rand = np.random.uniform(0,1)
        index = 4

        if rand <= pmax:
            index = index_pmax
        elif pmax < rand and rand <= pmax + phighermid:
            index = index_phighermid
        elif pmax + phighermid <rand and rand <= pmax + phighermid +plowermid:
            index = index_plowermid
        elif pmax + phighermid +plowermid < rand and rand <= 1:
            index = index_pmin
        return index

    def epsilon_greedy_step(self, state, epsilon):
        state_tensor = torch.tensor(state)
        network_prediction = self.dqn.q_network.forward(state_tensor)
        q_value, discrete_action_tensor = network_prediction.max(0)
        discrete_action = discrete_action_tensor.item()
        if self.epsilon_method == 'epsilon_greedy':
            epsilon_discrete_action = self.epsilon_shuffler(discrete_action,epsilon)  #Method 1 epsilon greedy
        elif self.epsilon_method == 'Q_epsilon_greedy':
            epsilon_discrete_action = self.Q_epsilon_shuffler(network_prediction,epsilon) #Method 2 epsilon greedy
        elif self.epsilon_method == 'Q_then_epsilon_greedy':
            qweight_discrete_action = self.Q_weighted_shuffler(network_prediction) #Method 3
            epsilon_discrete_action = self.epsilon_shuffler(qweight_discrete_action,epsilon)

        #now can't access environment so can only return the intended action
        # Store the action; this will be used later, when storing the transition
        self.action = epsilon_discrete_action
        continuous_action = self._discrete_action_to_continuous(int(epsilon_discrete_action))
        #return intended action
        return epsilon_discrete_action

    # Function to set the next state and distance, which resulted from applying action self.action at state self.state
    def set_next_state_and_distance(self, next_state, distance_to_goal):
        # Convert the distance to a reward
        reward = 1 - distance_to_goal
        ###different reward
        #reward = 1 - distance_to_goal**2
        #reward = - np.log(distance_to_goal)

        # Create a transition
        transition = (self.state, self.action, reward, next_state)
        # Now you can do something with this transition ...
        # Set the agent's state for the next step, as the next state from this step
        self.state = next_state
        # Update the agent's reward for this episode
        self.distance = distance_to_goal
        self.total_reward += reward
        ##add transition to collections deque
        self.dqn.replay_buffer.append(transition)
        ##add an entry for weight and prop
        self.dqn.replay_buffer.p_append(0)
        self.dqn.replay_buffer.w_append(self.dqn.replay_buffer.w_max)
        self.dqn.replay_buffer.update_p_deque()

        if self.replay_method == 'prioritised_experience':
            np.concatenate((self.dqn.deque_index,np.array([0])))

        ##sample mini batch
        if self.replay_method == 'random':
            mini_batch_transition = self.dqn.replay_buffer.sample_mini_batch(self.dqn.batch_size)
        elif self.replay_method == 'prioritised_experience':
            p_sample_index = self.dqn.replay_buffer.get_p_sample_index(self.dqn.batch_size)
            self.dqn.deque_index = p_sample_index
            mini_batch_transition = self.dqn.replay_buffer.p_sample_mini_batch(p_sample_index)
        ##batch train NN
        self.dqn_training_manager(mini_batch_transition) #loss = self.dqn_training_manager(mini_batch_transition)
            ###can't use this hueristic to see if stuck as not directly from distance or q network
            # if all(self.state == next_state):
            #     self.action = 4

    # Function to get the greedy action for a particular state
    def get_greedy_action(self, state):
        # # Here, the greedy action is fixed, but you should change it so that it returns the action with the highest Q-value
        # action = np.array([0.02, 0.0], dtype=np.float32)
        discrete_action = self.epsilon_greedy_step(state, 0)
        continuous_action = self._discrete_action_to_continuous(discrete_action)
        return continuous_action

# The DQN class determines how to train the above neural network.
class DQN:

    # The class initialisation function.
    def __init__(self, replay_method):
        #hyper parameters
        self.replay_method = replay_method
        self.batch_size = 500
        self.gamma = 0.9#0.98
        self.bellman_update_reward = 'bellman' # 'bellman' 'reward_only'
        self.use_target_network = 'use_target' # 'use_target' 'no_use_target'

        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = Network(input_dimension=2, output_dimension=4)
        self.dqn_target = Network(input_dimension=2, output_dimension=4)
        self.target_update() # initialise target dictionary and copy over the weights
        # Define the optimiser which is used when updating the Q-network. The learning rate determines how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(), lr=0.001)
        self.replay_buffer = ReplayBuffer()
        self.deque_index = collections.deque(maxlen=self.batch_size) #for prioritised experience selection
        self.deque_index.append(0)
        self.replay_epsilon = 0.01 #for prioritised experience selection

    # Function that is called whenever we want to train the Q-network. Each call to this function takes in a transition tuple containing the data we use to update the Q-network.
    def train_q_network(self, mini_batch_transition):
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss(mini_batch_transition)
        # Compute the gradients based on this loss, i.e. the gradients of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # Return the loss as a scalar
        return loss.item()

    # Function to calculate the loss for a particular transition.
    def _calculate_loss(self, mini_batch_transition):
        state_batch_list = []
        action_batch_list = []
        reward_batch_list = []
        for trans_tuple in mini_batch_transition:
            state, action, reward, next_state = trans_tuple
            state_batch_list.append(state)
            action_batch_list.append(action)
            reward_batch_list.append(reward)
        state_batch_tensor = torch.tensor(state_batch_list)
        action_batch_tensor = torch.tensor(action_batch_list)
        reward_batch_tensor = torch.tensor(reward_batch_list)
        network_prediction = self.q_network.forward(state_batch_tensor)
        target_network_prediction = self.dqn_target.forward(state_batch_tensor)
        unsqueeze_action_batch_tensor = torch.unsqueeze(action_batch_tensor, 0)
        unsqueeze_action_batch_tensor_transposed = unsqueeze_action_batch_tensor.t()
        cost = 0

        if self.bellman_update_reward == 'reward_only': # C = (R - Q)**2
            for i in range(self.batch_size):
                cost += (reward_batch_tensor[i] - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])**2
                #added for prioritised experience replay
                if self.replay_method == 'prioritised_experience':
                    self.replay_buffer.w_deque[self.deque_index[i]] = self.replay_epsilon + (reward_batch_tensor[i] - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])

        elif self.bellman_update_reward == 'bellman' and self.use_target_network == 'use_target': # C = (R + gamma maxQtarget(S')- Q(S))**2
            for i in range(self.batch_size):
                q_value, discrete_action_tensor = target_network_prediction[i].max(0)
                cost += (reward_batch_tensor[i] + self.gamma * q_value - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])**2
                #added for prioritised experience replay
                if self.replay_method == 'prioritised_experience':
                    self.replay_buffer.w_deque[self.deque_index[i]] = self.replay_epsilon + (reward_batch_tensor[i] + self.gamma * q_value - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])

        elif self.bellman_update_reward == 'bellman' and self.use_target_network == 'no_use_target': # C = (R + gamma maxQ(S')- Q(S))**2
            for i in range(self.batch_size):
                q_value, discrete_action_tensor = network_prediction[i].max(0)
                cost += (reward_batch_tensor[i] + self.gamma * q_value - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])**2
                #added for prioritised experience replay
                if self.replay_method == 'prioritised_experience':
                    self.replay_buffer.w_deque[self.deque_index[i]] = self.replay_epsilon + (reward_batch_tensor[i] + self.gamma * q_value - torch.gather(network_prediction,1,unsqueeze_action_batch_tensor_transposed)[i])

        cost = cost / self.batch_size
        return cost

    def target_update(self):
        state_dict = self.q_network.state_dict()
        self.dqn_target.load_state_dict(state_dict)

# The Network class inherits the torch.nn.Module class, which represents a neural network.
class Network(torch.nn.Module):

    # The class initialisation function. This takes as arguments the dimension of the network's input (i.e. the dimension of the state), and the dimension of the network's output (i.e. the dimension of the action).
    def __init__(self, input_dimension, output_dimension):
        # Call the initialisation function of the parent class.
        super(Network, self).__init__()
        # Define the network layers. This example network has two hidden layers, each with 100 units.
        self.layer_1 = torch.nn.Linear(in_features=input_dimension, out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.layer_3 = torch.nn.Linear(in_features=100, out_features=100)    #in case want extra dimesionality in the NN
        self.output_layer = torch.nn.Linear(in_features=100, out_features=output_dimension)

    # Function which sends some input data through the network and returns the network's output. In this example, a ReLU activation function is used for both hidden layers, but the output layer has no activation function (it is just a linear layer).
    def forward(self, input):
        layer_1_output = torch.nn.functional.relu(self.layer_1(input))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class ReplayBuffer:
    # The class initialisation function.
    def __init__(self):
        self.max_len = 20000
        self.collections_deque = collections.deque(maxlen=self.max_len)
        self.p_deque = collections.deque(maxlen=self.max_len)
        self.w_deque = collections.deque(maxlen=self.max_len)
        self.w_max = 1

    def append(self, transition):
        self.collections_deque.append(transition)
    def p_append(self, p):
        self.p_deque.append(p)
    def w_append(self, w):
        self.w_deque.append(w)

    def sample_mini_batch(self, batch_size):
        deque_size = len(self.collections_deque)
        return [self.collections_deque[x] for x in np.random.choice(deque_size, batch_size)]

    def update_p_deque(self):
        w_t = sum(self.w_deque)
        deque_size = len(self.collections_deque)
        for i in range(deque_size):
            self.p_deque[i] = self.w_deque[i] / w_t

    def get_p_sample_index(self, batch_size):
        deque_size = len(self.collections_deque)
        return np.random.choice(deque_size, batch_size, self.p_deque)

    def p_sample_mini_batch(self, p_sample_index):
        return [self.collections_deque[x] for x in p_sample_index]
