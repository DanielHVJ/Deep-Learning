import tensorflow as tf
import numpy as np

## https://awjuliani.medium.com/super-simple-reinforcement-learning-tutorial-part-1-fd544fab149



#List out our bandits. Currently bandit 4 (index#3) is set to most often provide a positive reward.
bandits = [4.2,-0.2, 0, -2.9]
num_bandits = len(bandits)
def pullBandit(bandit):
    #Get a random number.
    result = np.random.randn(1)
    if result > bandit:
        #return a positive reward.
        return 1
    else:
        #return a negative reward.
        return -1
    
bandits


from tensorflow.python.framework import ops
ops.reset_default_graph()
# tf.reset_default_graph()

import tensorflow.compat.v1 as v1
v1.disable_v2_behavior() 

#These two lines established the feed-forward part of the network. This does the actual choosing.
weights = tf.Variable(tf.ones([num_bandits]))
chosen_action = tf.argmax(weights,0)

#The next six lines establish the training proceedure. We feed the reward and chosen action into the network
#to compute the loss, and use it to update the network.
reward_holder = v1.placeholder(shape=[1],dtype=tf.float32)
action_holder = v1.placeholder(shape=[1],dtype=tf.int32)
responsible_weight = tf.slice(weights,action_holder,[1])
loss = -(v1.log(responsible_weight)*reward_holder)
optimizer = v1.train.GradientDescentOptimizer(learning_rate=0.001)
update = optimizer.minimize(loss)


total_episodes = 1000 #Set total number of episodes to train agent on.
total_reward = np.zeros(num_bandits) #Set scoreboard for bandits to 0.
e = 0.1 #Set the chance of taking a random action.

init = v1.initialize_all_variables()

# Launch the tensorflow graph
with v1.Session() as sess:
    sess.run(init)
    i = 0
    while i < total_episodes:
        
        #Choose either a random action or one from our network.
        if np.random.rand(1) < e:
            action = np.random.randint(num_bandits)
        else:
            action = sess.run(chosen_action)
        
        reward = pullBandit(bandits[action]) #Get our reward from picking one of the bandits.
        
        #Update the network.
        _,resp,ww = sess.run([update,responsible_weight,weights], feed_dict={reward_holder:[reward],action_holder:[action]})
        
        #Update our running tally of scores.
        total_reward[action] += reward
        if i % 50 == 0:
            print("Running reward for the " + str(num_bandits) + " bandits: " + str(total_reward))
        i+=1
print("The agent thinks bandit " + str(np.argmax(ww)+1) + " is the most promising....")
if np.argmax(ww) == np.argmax(-np.array(bandits)):
    print("...and it was rightget_ipython().getoutput("")")
else:
    print("...and it was wrongget_ipython().getoutput("")")



