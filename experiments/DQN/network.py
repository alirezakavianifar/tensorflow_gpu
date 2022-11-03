import tensorflow as tf
from collections import deque
from modified_tensorboard import ModifiedTensorBoard
import time
import numpy as np
import random

class DeepQNetwork:

    def __init__(self,
                 input_dims,
                 hidden_dims,
                 n_actions,
                 lr,
                 replay_memory_size=50_000,
                 min_replay_memory_size=10_000,
                 batch_size = 64,
                 model_name='DeepDelatIot'):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.lr = lr
        self.model_name = model_name
        self.replay_memory_size = replay_memory_size
        self.min_replay_memory_size = min_replay_memory_size
        self.batch_size = batch_size
        self.model = self.create_model()

        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=replay_memory_size)

        self.tensorboard = ModifiedTensorBoard(
            log_dir=f'logs\\{self.model_name}-{int(time.time())}')

    def create_model(self):
        inputs = tf.keras.Input(shape=(self.input_dims,))
        x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(inputs)
        output1 = tf.keras.layers.Dense(
            self.n_actions, activation='sigmoid', name='packetloss')(x)
        output2 = tf.keras.layers.Dense(
            self.n_actions, activation='sigmoid', name='latency')(x)

        model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        loss1 = tf.keras.losses.BinaryCrossentropy()
        loss2 = tf.keras.losses.BinaryCrossentropy()
        optim = tf.keras.optimizers.Adam()

        losses = {'packetloss': loss1, 'latency': loss2}
        # y = {'packetloss': y_train_p, 'latency': y_train_l}
        # y_test = {'packetloss': y_test_p, 'latency': y_test_l}

        model.compile(optimizer=optim, loss=losses, metrics=['accuracy'])

        return model

    def update_replay_memory(self, transition):
        self.replay_memory.append(transition)

    def get_qs(self, state, step):
        return self.model.predict(np.array(state))

    def train(self, terminal_state, step):
        if len(self.replay_memory) < self.min_replay_memory_size:
            return
        
        minibatch = random.sample(self.replay_memory, self.batch_size)
        
        current_states = np.array(transition[0] for transition in self.batch_size)
        current_qs_list = self.model.predict(current_states)
        
        new_current_states = np.array(transition[3] for transition in self.batch_size)
        future_qs_list = self.target_model.predict(new_current_states)
        
        
                                 
