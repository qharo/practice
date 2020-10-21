import tensorflow as tf
import tensorflow_probability as tfp

tfd = tfp.distributions
initial_distribution = tfd.Categorical(probs=[0.8, 0.2]) #the first day has a 0.8 chance of being cold, and 0.2 of hot

transition_distribution = tfd.Categorical(probs=[[0.7, 0.3], [0.2, 0.8]])
#a cold day has a 0.3 chance of being followed by a hot day, 0.7 of cold
#a hot day has a 0.2 chance of being followed by a cold day, 0.8 of hot

observation_distribution = tfd.Normal(loc=[0., 15.], scale=[5., 10.])
#on hot day, mean=15, sd = 10
#on cold day, mean = 0, sd = 5
 
#DEFINING THE MODEL
model = tfd.HiddenMarkovModel(
    initial_distribution = initial_distribution, 
    transition_distribution = transition_distribution,
    observation_distribution = observation_distribution,
    num_steps=7
)

mean = model.mean()
#mean is a partially defined tensor, so it has to be run within a session
#the new version, hence .compat.v1

with tf.compat.v1.Session() as sess:
    print(mean.numpy())