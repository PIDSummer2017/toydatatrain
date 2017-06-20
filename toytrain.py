#Tensorboard Modified Toytrain                                                 #2017 June 20                                                                                                                                                 
#IMPORT NECESSARY PACKAGES                                                    
import tensorflow as tf
import classification_image_gen
from classification_image_gen import make_classification_images as make_images

#START ACTIVE SESSION                                                         
sess = tf.InteractiveSession()

#DEFINE NECESSARY FUNCTIONS                                                   
def weight_variable(shape,name):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape,name):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def max_pool_1x1(x):
  return tf.nn.max_pool(x, ksize=[1, 1, 1, 1], strides=[1, 1, 1, 1], padding='SAME')

#PLACEHOLDERS                                                                 
x = tf.placeholder(tf.float32, [None, 784],name='x')
y_ = tf.placeholder(tf.float32, [None, 8],name='labels')

#RESHAPE IMAGE IF NEED BE                                                     
x_image = tf.reshape(x, [-1,28,28,1])
tf.summary.image('input',x_image,10)

#FIRST CONVOLUTIONAL LAYER                                                    
with tf.name_scope('conv1'):
  W_conv1 = weight_variable([5, 5, 1, 32],'W')
  b_conv1 = bias_variable([32],'b')
#Get an error saying histogram_summary doesn't exist...need to investigate!!!!  # W_conv1_hist = tf.histogram_summary('W_conv1',W_conv1)                       # b_conv1_hist = tf.histogram_summary('b_conv1',b_conv1) 
  h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
  h_pool1 = max_pool_2x2(h_conv1)

  tf.summary.histogram('weights',W_conv1)
  tf.summary.histogram('biases',b_conv1)
  tf.summary.histogram('activation',h_conv1)

#SECOND CONVOLUTIONAL LAYER 
with tf.name_scope('conv2'):
  W_conv2 = weight_variable([5, 5, 32, 64],'W')
  b_conv2 = bias_variable([64],'b')
  #W_conv2_hist = tf.histogram_summary('W_conv2',W_conv2)                        #b_conv2_hist = tf.histogram_summary('b_conv2',b_conv2)                                                   
  h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
  h_pool2 = max_pool_2x2(h_conv2)

  tf.summary.histogram('weights',W_conv2)
  tf.summary.histogram('biases',b_conv2)
  tf.summary.histogram('activation',h_conv2)

#FLATTENTING                                                                  
#I named it h_pool3_flat for conventions in case we decide to add 3rd convolutional layer. Make sure to comment this one out though if you do decide to use a 3rd convolutionl layer!!!!!  
h_pool3_flat = tf.reshape(h_pool2, [-1, 7*7*64])

#3RD CONVOLUTIONAL LAYER (OPTIONAL)                                           
#with tf.name_scope('conv3'):                                                    #W_conv3 = weight_variable([3, 3, 64, 128],'W')                                #b_conv3 = bias_variable([128],'b')                                                            
  #W_conv3_hist = tf.histogram_summary('W_conv3',W_conv3)                        #b_conv3_hist =\tf.histogram_summary('b_conv3',b_conv3)                                                         
  #h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)                      #h_pool3 = max_pool_1x1(h_conv3)                                                   
 # tf.summary.histogram('weights',W_conv3)                                      # tf.summary.histogram('biases',b_conv3)  
 # tf.summary.histogram('activation',h_conv3)                                                    
#FLATTENING                                                                    
#Only uncomment this line if you are going to add the 3rd convolutional layer and make sure to comment out line 69!!!                                        
#h_pool3_flat = tf.reshape(h_pool3, [-1, 7*7*128])                                               

#FINAL VARIABLE DEFINITIONS                                                    
with tf.name_scope('final'):
  W_final = weight_variable([7*7*64,8],'W')
  b_final = bias_variable([8],'b')
  #W_final_hist = tf.histogram_summary('W_final',W_final)                        #b_final_hist =\tf.histogram_summary('b_final',b_final)                                          
  y_conv = tf.matmul(h_pool3_flat, W_final) + b_final

#CROSS-ENTROPY                                                                 
with tf.name_scope('cross_entropy'):
  cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))

#TRAINING (RMS OR ADAM-OPTIMIZER OPTIONAL)                                     
with tf.name_scope('train'):
  train_step = tf.train.RMSPropOptimizer(0.0003).minimize(cross_entropy)

#ACCURACY                                                                      
with tf.name_scope('accuracy'):
  correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

  tf.summary.scalar('cross_entropy',cross_entropy)
  tf.summary.scalar('accuracy', accuracy)

saver= tf.train.Saver()

sess.run(tf.global_variables_initializer())

#MERGE SUMMARIES FOR TENSORBOARD                                               
merged_summary=tf.summary.merge_all()

#WRITE SUMMARIES TO LOG DIRECTORY LOGS6                                        
writer=tf.summary.FileWriter("./logs6")
writer.add_graph(sess.graph)

#MAKE IMAGES AND TRAINING                                                      
for _ in range(1000):
    batch = make_images()
    #train_step.run(feed_dict = {x:batch[0], y_:batch[1]})                                       
#TRAINING                                                                     
for i in range(1000):

  batch = make_images()

  if i%100 == 0:
    
    s = sess.run(merged_summary, feed_dict={x:batch[0], y_:batch[1]})
    writer.add_summary(s,i)

    train_accuracy = accuracy.eval(feed_dict={x:batch[0], y_: batch[1]})

    print("step %d, training accuracy %g"%(i, train_accuracy))
  train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  #sess.run(train_step,feed_dict={x: batch[0], y_: batch[1]})                                    
batchtest = make_images(1000)
if i%1000 ==0:
    test_accuracy = accuracy.eval(feed_dict={x:batchtest[0], y_:batchtest[1]})
    print("step %d, test accuracy %g"%(i, test_accuracy))

batch = make_images(1000)

print("test accuracy %g"%accuracy.eval(feed_dict={x: batch[0], y_: batch[1]}))
print('Run `tensorboard --logdir=logs6` in terminal to see the results.')


