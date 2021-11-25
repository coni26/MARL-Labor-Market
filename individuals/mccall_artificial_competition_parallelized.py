import numpy as np
import matplotlib.pyplot as plt
import collections
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers 
import quantecon as qe
from quantecon.distributions import BetaBinomial
plt.rcParams["figure.figsize"] = (11, 5)


### In this version, the training is parallelized to increase the speed.
### Indeed, one of the time-consuming steps is to choose the action of the agents.
### We could have a single agent, but in this case we would create less training data than if we had several agents.
### So, with several agents, we compute their decision at the same time, which does not increase the computation time, 
### but allows to multiply the data.

##################
### Parameters ###
##################

unemployment_salary = 2
gamma = 0.90

application_cost = 0.5

γ = 0.5 # Competition parameter, probability of acceptance

n = 300                                 
w_min, w_max = 2, 5
w_default = np.linspace(w_min, w_max, n+1)      
a, b = 2.5, 2
q_default = BetaBinomial(n, a, b).pdf()

#plt.plot(w_default, q_default, 'b+') # uncomment to see distribution

def u(c, σ=1.3):
    return (c**(1 - σ) - 1) / (1 - σ)
  
class Agent:
    
    def __init__(self, unemployment_salary):
        #self.prod = np.random.choice(w_default, p=q_default) # Choice of productivity distribution
        self.prod = 2.5 + 2.5 * np.random.rand() # For better training, uniform distribution
        self.prev_prod = self.prod # We have to keep previous productivity for historization
        self.state = 0 
        self.prev_state = 0
        self.salary = unemployment_salary # We have to keep previous state for historization
        self.prev_salary = unemployment_salary # We have to keep previous salary for historization
        
        
class environment:
    
    def __init__(self, unemployment_salary, dec_eps = 0.99997, γ=γ):
        self.time = 0
        self.end = 400 # Number of frames in one episode 
        self.eps = 1
        self.min_eps = 0.1
        self.dec_eps = dec_eps
        self.salaries = None
        self.unemployment_salary = unemployment_salary
        self.γ = γ

    def step(self, choices, l_agents):
        self.time += 1
  
        rewards = [0] * len(l_agents)
        
        for i, agent in enumerate(l_agents):
            agent.prev_state = agent.state # We keep in memory previous parameters
            agent.prev_prod = agent.prod
            agent.prev_salary = agent.salary
            if agent.state: # Agent had a job
                if choices[i]: # He keep his job
                    rewards[i] = u(agent.salary)
                    agent.prod += 0.01 # His productivity increases
                    agent.prod = min(5, agent.prod) # Productivity is capped
                else: # He leaves his job
                    rewards[i] = u(unemployment_salary)
                    agent.state = 0
                    agent.salary = unemployment_salary
            else: # Agent was unemployed
                if choices[i]: 
                    if agent.prod >= self.salaries[i] and np.random.rand() < self.γ: # He gets the job
                        rewards[i] = u(self.salaries[i] - application_cost)
                        agent.salary = self.salaries[i] 
                        agent.state = 1 
                    else: # He is rejected
                        rewards[i] = u(self.unemployment_salary - application_cost)
                        agent.prod -= 0.01 # His productivity decreases
                        agent.prod = max(2, agent.prod) # Productivity is capped
                else: # He did not apply
                    rewards[i] = u(self.unemployment_salary)
                    agent.prod -= 0.01
                    agent.prod = max(2, agent.prod)
            
        self.eps = self.eps * self.dec_eps
        self.eps = max(self.min_eps, self.eps)
        
        self.salaries = np.random.choice(w_default, p=q_default, size = nb_agents) # We draw the different salaries proposed for all agents, even if they are employed
            
        return self.salaries, rewards, self.time >= self.end
    
    def reset(self):
        self.time = 0
        self.salaries = np.random.choice(w_default, p=q_default, size = nb_agents)
        return self.salaries
        
    
      
      
######################
### Neural network ###
######################
num_actions = 2

def create_model_q():
    inputs = layers.Input(shape=(3,))
    
    layer0 = layers.Dense(16, activation="linear", name='layer0')(inputs)
    layer01 = layers.Dense(16, activation="sigmoid", name='layer01')(layer0)
    action_0 = layers.Dense(1, activation="linear", name='action_0')(layer01)
    
    layer1 = layers.Dense(16, activation="linear", name='layer1')(inputs)
    layer2 = layers.Dense(16, activation="sigmoid", name='layer2')(layer1)
    layer3 = layers.Dense(16, activation="sigmoid", name='layer3')(layer2)
    action_1 = layers.Dense(1, activation="linear", name='action_1')(layer3)
    
    action = layers.Concatenate(name='action')([action_0, action_1])
 
    return keras.Model(inputs=inputs, outputs=action)

def create_model_s():
    inputs = layers.Input(shape=(3,))
    
    layer0 = layers.Dense(16, activation="linear", name='layer0')(inputs)
    layer01 = layers.Dense(16, activation="sigmoid", name='layer01')(layer0)
    action_0 = layers.Dense(1, activation="linear", name='action_0')(layer01)
    
    layer1 = layers.Dense(16, activation="linear", name='layer1')(inputs)
    layer2 = layers.Dense(16, activation="sigmoid", name='layer2')(layer1)
    layer3 = layers.Dense(16, activation="sigmoid", name='layer3')(layer2)
    action_1 = layers.Dense(1, activation="linear", name='action_1')(layer3)
    
    action = layers.Concatenate(name='action')([action_0, action_1])

    return keras.Model(inputs=inputs, outputs=action)
  

  
  
@tf.function
def maj_q(state_sample, updated_q_values, action_sample):
    masks = tf.one_hot(action_sample, num_actions)
    with tf.GradientTape() as tape:
        q_values = model_q(state_sample, training=True)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
    
    
    grads = tape.gradient(loss, model_q.trainable_variables)
    optimizer_q.apply_gradients(zip(grads, model_q.trainable_variables))

@tf.function
def maj_s(state_sample, updated_q_values, action_sample):
    masks = tf.one_hot(action_sample, num_actions)
    
    with tf.GradientTape() as tape:
        q_values = model_s(state_sample, training=True)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
    
    grads = tape.gradient(loss, model_s.trainable_variables)
    optimizer_s.apply_gradients(zip(grads, model_s.trainable_variables))
    
    
######################    
### Model training ###
######################

nb_agents = 10 # Number of parallelized agents


model_q = create_model_q()
model_s = create_model_s()

model_q_target = create_model_q()
model_s_target = create_model_s()

update_target = 1

optimizer_q = keras.optimizers.Adam(learning_rate=0.001)
optimizer_s = keras.optimizers.Adam(learning_rate=0.001)
loss_function = keras.losses.MeanSquaredError()

env = environment(unemployment_salary)
env.dec_eps = 0.999992

batch_size = 512
update_after_actions = 1 # We take 1, because in one frame, we learn 10 datas. We could also take more than 1.

action_history_q = []
state_history_q = []
state_next_history_q = []
rewards_history_q = []
agent_state_history_q = []

action_history_s = []
state_history_s = []
state_next_history_s = []
rewards_history_s = []
agent_state_history_s = []

max_memory_length = 10000

frame_count = 0

for ep in range(500):
        
    l_agents = [Agent(unemployment_salary) for i in range(nb_agents)]
    
    done = False
    salaries = env.reset()
    frame_count = 0

    while not done:
        
        frame_count += 1
        
        arr_agent_q = np.zeros((nb_agents, 3))
        arr_agent_s = np.zeros((nb_agents, 3))
        arr_state = np.zeros(nb_agents)
        for i, agent in enumerate(l_agents):
            arr_agent_q[i, :] = [agent.prod, agent.unemployment_salary, agent.salary]
            arr_agent_s[i, :] = [agent.prod, agent.unemployment_salary, salaries[i]]
            arr_state[i] = agent.state

        state_tensor_q = tf.convert_to_tensor(arr_agent_q)
        state_tensor_s = tf.convert_to_tensor(arr_agent_s)
        actions = (np.array([arr_state, arr_state]).T * model_q(state_tensor_q).numpy() + np.array([1 - arr_state, 1 - arr_state]).T * model_s(state_tensor_s).numpy()).argmax(axis=1)

        n_changes = np.random.binomial(nb_agents, env.eps)
        actions[np.random.randint(0, nb_agents, size=n_changes)] = np.random.randint(2, size=n_changes)
        
        next_salaries, rewards, done = env.step(actions, l_agents)
        
        for i, agent in enumerate(l_agents): # Historization
            if agent.prev_state: # Agent was employed, so his choice wan on Q
                action_history_q.append(actions[i])
                state_history_q.append([agent.prev_prod, unemployment_salary, agent.prev_salary])
                if actions[i]: # He leaves is job
                    state_next_history_q.append([agent.prod, unemployment_salary, agent.salary])
                else:
                    state_next_history_q.append([agent.prod, unemployment_salary, next_salaries[i]])
                rewards_history_q.append(rewards[i])
            else: # Agent was unemployed, so his choice wan on S
                action_history_s.append(actions[i])
                state_history_s.append([agent.prod, unemployment_salary, salaries[i]])
                if agent.state: # If he is accepted
                    agent_state_history_s.append(agent.state)
                    state_next_history_s.append([agent.prod, unemployment_salary, agent.salary])
                else:
                    agent_state_history_s.append(agent.state)
                    state_next_history_s.append([agent.prod, unemployment_salary, next_salaries[i]])
                rewards_history_s.append(rewards[i])


        ### model_q update
        if frame_count % update_after_actions == 0 and len(action_history_q) > batch_size:
            
            indices = np.random.choice(range(len(action_history_q)), size=batch_size)

            state_sample = np.array([state_history_q[i] for i in indices])
            state_next_sample = np.array([state_next_history_q[i] for i in indices])
            rewards_sample = [rewards_history_q[i] for i in indices]
            action_sample = [action_history_q[i] for i in indices]
 
            state_next_tensor = tf.convert_to_tensor(state_next_sample)
            action_sample = np.array(action_sample)
            future_reward = np.array([action_sample, action_sample]).T * model_q_target(state_next_tensor).numpy() + np.array([1 - action_sample, 1- action_sample]).T * model_s_target(state_next_tensor).numpy()
            updated_q_values = rewards_sample + gamma * np.max(future_reward, axis=1)
            
            maj_q(state_sample, updated_q_values, action_sample)
            
        ### model_s update
        if frame_count % update_after_actions == 0 and len(action_history_s) > batch_size:
            
            indices = np.random.choice(range(len(action_history_s)), size=batch_size)

            state_sample = np.array([state_history_s[i] for i in indices])
            state_next_sample = np.array([state_next_history_s[i] for i in indices])
            rewards_sample = [rewards_history_s[i] for i in indices]
            action_sample = [action_history_s[i] for i in indices]
            agent_state_sample = [agent_state_history_s[i] for i in indices]
            
            state_next_tensor = tf.convert_to_tensor(state_next_sample)
            agent_state_sample = np.array(agent_state_sample)
            future_reward = np.array([agent_state_sample, agent_state_sample]).T * model_q_target(state_next_tensor).numpy() + np.array([1 - agent_state_sample, 1- agent_state_sample]).T * model_s_target(state_next_tensor).numpy()
            updated_q_values = rewards_sample + gamma * np.max(future_reward, axis=1)
            
            maj_s(state_sample, updated_q_values, action_sample)

            
        salaries = next_salaries
    
    
    if len(state_next_history_q) > max_memory_length:
        agent_state_history_q = agent_state_history_q[-max_memory_length:]
        action_history_q = action_history_q[-max_memory_length:]
        state_next_history_q = state_next_history_q[-max_memory_length:]
        state_history_q = state_history_q[-max_memory_length:]
        rewards_history_q = rewards_history_q[-max_memory_length:]

    if len(state_next_history_s) > max_memory_length:
        agent_state_history_s = agent_state_history_s[-max_memory_length:]
        action_history_s = action_history_s[-max_memory_length:]
        state_next_history_s = state_next_history_s[-max_memory_length:]
        state_history_s = state_history_s[-max_memory_length:]
        rewards_history_s = rewards_history_s[-max_memory_length:]
        
    if ep % update_target == 0:
        model_q_target.set_weights(model_q.get_weights())
        model_s_target.set_weights(model_s.get_weights())
    
    if ep % 5 == 0: # Just visualization, not necessary
        print('_____________________________',ep,'_____________________________', env.eps)
        for prod in [3.5]:
            state_tensor = tf.convert_to_tensor(np.array([[prod] * len(w_default), [unemployment_salary] * len(w_default), w_default]).T)
            action_probs = model_q(state_tensor, training=False)
            l_deny_q = action_probs[:, 0]
            l_accept_q = action_probs[:, 1]
            action_probs = model_s(state_tensor, training=False)
            l_deny_s = action_probs[:, 0]
            l_accept_s = action_probs[:, 1]
            plt.plot(w_default, l_deny_q, label='Q_0__' + str(prod))
            plt.plot(w_default, l_accept_q, label='Q_1__' + str(prod))
            plt.plot(w_default, l_deny_s, label='S_0__' + str(prod))
            plt.plot(w_default, l_accept_s, label='S_1__' + str(prod))
            plt.axvline(prod)
        plt.plot(w_default, u(w_default) / (1 - gamma), label='theory')
        plt.legend()
        plt.show()
      




