unemployment_salary = 2
gamma = 0.95

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
        self.prod = np.random.choice(w_default, p=q_default)  
        self.state = 0
        self.salary = unemployment_salary
        
        
class environment:
    
    def __init__(self, unemployment_salary, dec_eps = 0.99997):
        self.time = 0
        self.end = 50
        self.eps = 1
        self.min_eps = 0.1
        self.dec_eps = dec_eps
        self.salary = None
        self.unemployment_salary = unemployment_salary
    
    def step(self, choice, agent):
        self.time += 1
        if choice:
            reward = self.salary
            agent.state = 1
        else: 
            reward = self.unemployment_salary
            self.salary = np.random.choice(w_default, p=q_default)    
            
        self.eps = self.eps * self.dec_eps
        self.eps = max(self.min_eps, self.eps)
            
        return self.salary, u(reward), self.time >= self.end or agent.state
    
    def reset(self):
        self.time = 0
        self.salary = np.random.choice(w_default, p=q_default)
        return self.salary
      
      
######################
### Neural network ###
######################
        
num_actions = 2

def create_model():
    inputs = layers.Input(shape=(3,))
    
    layer0 = layers.Dense(16, activation="sigmoid", name='layer0')(inputs)
    layer01 = layers.Dense(16, activation="sigmoid", name='layer01')(layer0)
    action_0 = layers.Dense(1, activation="linear", name='action_0')(layer01)
    
    layer1 = layers.Dense(16, activation="linear", name='layer1')(inputs)
    layer2 = layers.Dense(16, activation="linear", name='layer2')(layer1)
    layer3 = layers.Dense(16, activation="sigmoid", name='layer3')(layer2)
    action_1 = layers.Dense(1, activation="linear", name='action_1')(layer3)
    
    action = layers.Concatenate(name='action')([action_0, action_1])

    return keras.Model(inputs=inputs, outputs=action)
  
  

  
def update_gamma(gamma):
    return (gamma - 0.945) * 100
  
def downdate_gamma(gamma):
    return gamma / 100 + 0.945
  
  
@tf.function
def maj_model(action_sample, state_sample, updated_q_values):
    masks = tf.one_hot(action_sample, num_actions)

    with tf.GradientTape() as tape:
        q_values = model(state_sample)
        q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        loss = loss_function(updated_q_values, q_action)
    
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))  
    
    
### entrainement
j_ = 0

model = create_model()
model_target = create_model()

update_target = 100

optimizer = keras.optimizers.Adam(learning_rate=0.001)
loss_function = keras.losses.MeanSquaredError()

env = environment(unemployment_salary)
env.dec_eps = 0.999998

sum_grads = None

batch_size = 1024
update_after_actions = 4

action_history = []
state_history = []
state_next_history = []
rewards_history = []
done_history = []

frame_count = 0

max_memory_length = 10000

c_vals = np.linspace(1,3,21)

prec = time.time()
for ep in range(1000000):
    gamma = np.random.rand() * 0.117 + 0.875
    unemployment_salary = np.random.rand() * 2.5 + 0.8
    
    if ep % 5000 == 0:
        print('________________________________________________________________')
        print('_____________________________',ep,'_____________________________', env.eps)
        print('Temps :', time.time() - prec)
        prec = time.time()
        
        
    cum_rewards = 0
    cum_loss = 0
    agent = Agent(unemployment_salary)
    
    done = False
    salary = env.reset()
    env.unemployment_salary = unemployment_salary

    while not done:
        
        frame_count += 1
         
        state_tensor = tf.convert_to_tensor([update_gamma(gamma), salary, unemployment_salary])
        state_tensor = tf.expand_dims(state_tensor, 0)
        if env.eps > np.random.rand():
            action = np.random.randint(2)
        else:
            action = tf.argmax(model(state_tensor)[0]).numpy()
        
        next_salary, reward, done = env.step(action, agent)
        
        action_history.append(action)
        state_history.append([update_gamma(gamma), salary, unemployment_salary])
        state_next_history.append([update_gamma(gamma), next_salary, unemployment_salary])
        rewards_history.append(reward)
        done_history.append(done)
        
        
        
        if frame_count % update_after_actions == 0 and len(action_history) > batch_size:

            indices = np.random.randint(len(action_history), size=batch_size)

            state_sample = np.array([state_history[i] for i in indices])
            state_next_sample = np.array([state_next_history[i] for i in indices])
            rewards_sample = np.array([rewards_history[i] for i in indices])
            action_sample = [action_history[i] for i in indices]
            done_sample = np.array([done_history[i] for i in indices])
            
            action_sample = tf.convert_to_tensor(action_sample)
            state_sample = tf.convert_to_tensor(state_sample)
            _gamma = downdate_gamma(np.array(state_next_sample[:, 0]))
            updated_q_values = rewards_sample / (1 - _gamma) * done_sample + (1 - np.array(done_sample)) * (rewards_sample + _gamma * tf.reduce_max(model_target(state_next_sample), axis = 1))

            maj_model(action_sample, state_sample, updated_q_values)
            

        salary = next_salary
    if len(done_history) > max_memory_length:
        done_history = done_history[-max_memory_length:]
        action_history = action_history[-max_memory_length:]
        rewards_history = rewards_history[-max_memory_length:]
        state_next_history = state_next_history[-max_memory_length:]
        state_history = state_history[-max_memory_length:]
        
    if ep % update_target == 0:
        model_target.set_weights(model.get_weights())
        
    if ep%5000 == 0:
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17,5))
        state_tensor = tf.convert_to_tensor(np.array([[update_gamma(0.90)] * len(w_default), w_default, [2] * len(w_default)]).T)
        action_probs = model_target(state_tensor, training=False)
        l_deny = action_probs[:, 0]
        l_accept = action_probs[:, 1]
        ax1.plot(w_default, l_deny, label='Reject')
        ax1.plot(w_default, l_accept, label='Accept')
        ax1.plot(w_default, u(w_default) * 10)
        ax1.legend()
        state_tensor = tf.convert_to_tensor(np.array([[update_gamma(0.95)] * len(w_default), w_default, [2] * len(w_default)]).T)
        action_probs = model_target(state_tensor, training=False)
        l_deny = action_probs[:, 0]
        l_accept = action_probs[:, 1]
        ax2.plot(w_default, l_deny, label='Reject')
        ax2.plot(w_default, l_accept, label='Accept')
        ax2.plot(w_default, u(w_default) * 20)
        ax2.legend()
        state_tensor = tf.convert_to_tensor(np.array([[update_gamma(0.99)] * len(w_default), w_default, [2] * len(w_default)]).T)
        action_probs = model_target(state_tensor, training=False)
        l_deny = action_probs[:, 0]
        l_accept = action_probs[:, 1]
        ax3.plot(w_default, l_deny, label='Reject')
        ax3.plot(w_default, l_accept, label='Accept')
        ax3.plot(w_default, u(w_default) * 100)
        ax3.legend()
        plt.show()

        


