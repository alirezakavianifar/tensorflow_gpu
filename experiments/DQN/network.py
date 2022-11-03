import tensorflow as tf
from collections import deque


class DeepQNetwork:

    def __init__(self, input_dims, hidden_dims, n_actions, lr, replay_memory_size=10_000):
        self.input_dims = input_dims
        self.hidden_dims = hidden_dims
        self.n_actions = n_actions
        self.lr = lr
        self.replay_memory_size = replay_memory_size
        self.model = self.create_model()
        
        
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())
        
        self.replay_memory = deque(maxlen=replay_memory_size)


    def create_model(self):
        inputs = tf.keras.Input(shape=(self.input_dims,))
        x = tf.keras.layers.Dense(128, activation=tf.nn.relu)(inputs)
        output1 = tf.keras.layers.Dense(
            self.n_actions, activation='sigmoid', name='packetloss')(x)
        output2 = tf.keras.layers.Dense(self.n_actions, activation='sigmoid', name='latency')(x)

        model = tf.keras.Model(inputs=inputs, outputs=[output1, output2])
        loss1 = tf.keras.losses.BinaryCrossentropy()
        loss2 = tf.keras.losses.BinaryCrossentropy()
        optim = tf.keras.optimizers.Adam()

        losses = {'packetloss': loss1, 'latency': loss2}
        # y = {'packetloss': y_train_p, 'latency': y_train_l}
        # y_test = {'packetloss': y_test_p, 'latency': y_test_l}

        model.compile(optimizer=optim, loss=losses, metrics=['accuracy'])
        
        return model
