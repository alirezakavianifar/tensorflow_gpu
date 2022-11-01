import tensorflow as tf


# class DeepQNetwork(tf.keras.Model):

#     def __init__(self, input_dims, hidden_dims, n_actions, lr):
#         super(DeepQNetwork, self).__init__()
#         self.input_dims = input_dims
#         self.n_actions = n_actions
#         self.lr = lr
#         self.hidden_dims = hidden_dims
#         self.inputs = tf.keras.Input(shape=(self.input_dims,))
#         self.dense1 = tf.keras.layers.Dense(self.hidden_dims, activation=tf.nn.relu)
#         self.dense2 = tf.keras.layers.Dense(self.hidden_dims, activation=tf.nn.relu)
#         self.output1 = tf.keras.layers.Dense(self.n_actions, activation=tf.nn.softmax)

#     def call(self):
#         x = self.dense1(self.inputs)
#         x = self.dense2(x)
#         return self.output1(x)

# model = DeepQNetwork(64, 64, 2, 1e-3)

# model.build((1,64))

def create_model(input_dims, n_actions=10, lr=1e-3, from_mnist=False):
    if from_mnist:
        inputs = tf.keras.Input(shape=(28, 28))
        flatten = tf.keras.layers.Flatten()
        loss1 = tf.keras.losses.SparseCategoricalCrossentropy()
        optim = tf.keras.optimizers.Adam(learning_rate=lr)
        name = 'mnist'
    else:
        inputs = tf.keras.Input(shape=(input_dims,))
        name = 'energyconsumption'
        loss1 = tf.keras.losses.CategoricalCrossentropy()
        optim = tf.keras.optimizers.SGD(learning_rate=lr)

    dense1 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    dense2 = tf.keras.layers.Dense(64, activation=tf.nn.relu)
    output1 = tf.keras.layers.Dense(
        n_actions, activation=tf.nn.softmax, name=name)
    # output2 = tf.keras.layers.Dense(
    #     n_actions, activation=tf.nn.softmax, name='packetloss')

    x = flatten(inputs)
    x = dense1(x)
    x = dense2(x)
    output = output1(x)
    outputs = [output]
    model = tf.keras.Model(inputs=inputs, outputs=output)

    # loss2 = tf.keras.losses.BinaryCrossentropy()

    metrics = [tf.keras.metrics.Accuracy()]
    losses = {name: loss1}
    #   'packetloss': loss2

    model.compile(loss=loss1, optimizer=optim, metrics=metrics)
    return model


# model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.lr),
#                 loss=tf.keras.losses.categorical_crossentropy,
#                 metrics=[tf.keras.metrics.Accuracy()])
