
# coding: utf-8

# In[90]:

import math
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
from tensorflow.python.framework import ops
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



        
def gen_sets_and_label(x_items, x_dim, y_dim):
    x_data = (np.random.random([x_items,x_dim])-0.5) *2
    x_data=x_data.astype(np.float32)
    y_data = np.random.random([x_items,y_dim]).astype(np.float32)
    y_data = y_data - y_data
#     y_data=y_data.astype(np.float32)
    for i, v in enumerate(x_data):
        dist = np.sqrt(v[0]**2 + v[-1]**2)
        y_data[i][0] = dist
    
    y_data_first_col = y_data[:,0]
    
    low = np.min(y_data_first_col)
    
    delta = (np.max(y_data_first_col) - np.min(y_data_first_col)) / (y_dim + 1.0)
    d_class_board = delta / 10.0
    left_0  = low + 0.0 * delta + d_class_board
    right_0 = low + 1.0 * delta - d_class_board
    
    left_1  = low + 1.0 * delta + d_class_board
    right_1 = low + 2.0 * delta - d_class_board
    
    left_2  = low + 2.0 * delta + d_class_board
    right_2 = low + 3.0 * delta - d_class_board
    
    for a in range(x_items):
        
        if   left_0 <= y_data[a][0] <= right_0:
            y_data[a][0] = 0.0
            y_data[a][1-1] = 1.0
            
        elif left_1 <= y_data[a][0] <= right_1:
            y_data[a][0] = 0.0
            y_data[a][2-1] = 1.0
            
        elif left_2 <= y_data[a][0] <= right_2 :
            y_data[a][0] = 0.0
            y_data[a][3-1] = 1.0
        else:
            y_data[a][0] = 0.0
        
    delete_idx_list = []
    for i,v in enumerate(y_data):
        if np.abs(np.sum(v) - 0) < 1e-2:
            delete_idx_list.append(i)

    x_data = np.delete(x_data, delete_idx_list, axis=0)
    y_data = np.delete(y_data, delete_idx_list, axis=0) 
    return [x_data, y_data]


class ML(object):
    def __init__(self, x_dim ,y_dim ,OBJ_ML_CONST):
        
        x = tf.placeholder( tf.float32, [None,x_dim] )
        y = tf.placeholder( tf.float32, [None,y_dim] )
        keep_prob = tf.placeholder(tf.float32)
        
        sz = int ( math.sqrt( float( x_dim ) )  ) 
        
#         img = tf.reshape(x, [-1,x_dim,1,1])
        img = x
#         img = tf.reshape(x, [-1,sz,sz,1])
        self.x = x
        self.y = y
        self.img = img 
        # OBJ_ML_CONST
        self.lr = OBJ_ML_CONST["lr"]
        self.iter_times = OBJ_ML_CONST["iter_times"]
        self.batch_size = OBJ_ML_CONST["batch_size"]
        self.keep_prob_rate = OBJ_ML_CONST["keep_prob_rate"]
        self.accu_percent = OBJ_ML_CONST["accu_percent"]
        self.id_train = OBJ_ML_CONST["id_train"]
        
        self.flat_ele = "FLAT_ELE INIT"
        self.keep_prob = keep_prob 

    
    def say(self):
        print ("- self.x: ", self.x)
        print ("- self.y: ", self.y)
        print ("- self.img: ", self.img)
        
        print("- self.lr: %g" % self.lr)
        print("- self.iter_times: ", self.iter_times)
        print("- self.batch_size: ", self.batch_size)
        print("- self.keep_prob_rate: ", self.keep_prob_rate)
        
        print("- self.accu_percent: ", self.accu_percent)
#         print ("- self.id_train: ", self.id_train)
        return self
        
    

    def tf_w_b_by_k(self, k ):
        w_ = tf.truncated_normal(k, stddev=0.1)
        w = tf.Variable(w_)
        b_ = tf.constant(0.1, shape=k[-1:])
        b = tf.Variable(b_)
        return [w,b]
    
    def conv2d_by_k( self, k, act_fun="relu" ):
        [w,b] = self.tf_w_b_by_k(k)
        id_conv_ = tf.nn.conv2d(self.img,w, strides=[1,1,1,1], padding='SAME')
        id_conv = tf.nn.relu( id_conv_ + b )
        self.img = id_conv 
        return self
    
    
    def pool_2x2(self):
        id_p = tf.nn.max_pool( self.img, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME' )
        self.img = id_p
        return self

    def cal_w_b_by_k(self, k, act_fun="relu"):
        [w,b] = self.tf_w_b_by_k(k)
        h_ = tf.matmul(self.img, w) + b
        if act_fun == "relu" :
            h = tf.nn.relu(h_)
            self.img = h 
            return self
        if act_fun == "relu6" :
            h = tf.nn.relu6(h_)
            self.img = h 
            return self        
        if act_fun == "softmax" :
            h = tf.nn.softmax(h_)
            self.img = h 
            return self
        if act_fun == "None":
            self.img = h_
            return self
        raise  ValueError('- act_fun is "relu" or "softmax"! ')
    
   
        
    def tf_dropout(self, keep_prob_):
        #keep_prob is NOT a number
        hh = tf.nn.dropout(self.img, keep_prob_)
        self.img = hh 
        return self
    
    def loss_fun(self,str_type="entropy"):
        a_y = self.y
        a_y_ = self.img  
        self.y_ = self.img
        if str_type == "entropy":
            loss = tf.reduce_mean( -1*tf.reduce_sum( a_y*tf.log(a_y_) , reduction_indices=[1] ) )
            self.loss = loss
            #return tf.reduce_mean( -1*tf.reduce_sum( y*tf.log(y_) , reduction_indices=[1] ) )
        else:
            raise  ValueError('- only entropy can be used!')
    

    def set_train_args(self):
        if self.loss == "LOSS INIT":
            raise  ValueError('- please set loss func')
        else:
            id_train = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
            self.id_train = id_train
            return id_train

    def cal_accu_percent(self):
        a_y = self.y
        a_y_ = self.img
        vec_0_1 = tf.equal (tf.argmax(a_y,1),tf.argmax(a_y_,1)) 
        accu_percent = tf.reduce_mean( tf.cast(vec_0_1,tf.float32) )
        self.accu_percent = accu_percent
        return self.accu_percent


    def start_train_loop(self,x_data, y_data, x_test, y_test):
        cnt_net_ok = 0
        batch_size = self.batch_size
        a_rate_e = 0.0
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            for i in range(self.iter_times):
                s_i = np.random.randint(0,x_data.shape[0]-batch_size)
                bd_0 =   x_data[s_i:s_i+batch_size] 
                bd_1 =   y_data[s_i:s_i+batch_size] 
                
                if self.accu_percent == "ACCU_PERCENT INIT":
                    raise  ValueError('- please set accu_percent : cal_accu_percent(y,y_)')

                if self.id_train == "ID_TRAIN INIT":
                    raise  ValueError('- please set id_train : set_train_args()')

                if i%100 == 0:
                    a_rate_e = self.accu_percent.eval( feed_dict = {self.x:bd_0, self.y:bd_1, self.keep_prob: 1.0} )  # y_ and y
                    print ("- train %d, accuracy: %g" %(i,a_rate_e))
                    if a_rate_e > 0.971:
                        cnt_net_ok += 1
                    else:
                        cnt_net_ok = 0
                    if cnt_net_ok == 3:
                        tf.train.Saver().save(sess, "./mm/mm.ckpt")
                        print ("- save mm.ckpt")
                        break                      
                   
                self.id_train.run( feed_dict = {self.x:bd_0, self.y:bd_1, self.keep_prob: self.keep_prob_rate } ) 

                    
#             feed_content = {self.x:mnist.test.images,self.y:mnist.test.labels,self.keep_prob:1.0}
            x_test_accu_precent = self.accu_percent.eval( feed_dict={self.x: x_test, self.y:y_test, self.keep_prob:1.0} )
            print ("- the test accuracy %g " % x_test_accu_precent )
            if cnt_net_ok != 3 and x_test_accu_precent > 0.95:
                tf.train.Saver().save(sess, "./mm/mm.ckpt")
                print ("- save mm.ckpt")

                
            
            
        return self
    
    def restart_train_loop(self,x_data, y_data, x_test, y_test, initial_accu_percent):
        cnt_net_ok = 0
        batch_size = self.batch_size
        self.lr /= 3.0
#         self.keep_prob_rate *= 0.9
#         self.keep_prob_rate *= 1.1
        self.id_train = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        if 1:
#             sess.run(tf.global_variables_initializer())
            self.say()
            print ("\n- restart trainning process")
            for i in range(self.iter_times):
                s_i = np.random.randint(0,x_data.shape[0]-batch_size)
                bd_0 =   x_data[s_i:s_i+batch_size] 
                bd_1 =   y_data[s_i:s_i+batch_size] 
                
                if self.accu_percent == "ACCU_PERCENT INIT":
                    raise  ValueError('- please set accu_percent : cal_accu_percent(y,y_)')

                if self.id_train == "ID_TRAIN INIT":
                    raise  ValueError('- please set id_train : set_train_args()')

                if i%100 == 0:
                    a_rate_e = self.accu_percent.eval( feed_dict = {self.x:bd_0, self.y:bd_1, self.keep_prob: 1.0} )  # y_ and y
                    print ("- train %d, accuracy: %g" %(i,a_rate_e))
                    if a_rate_e > 0.985:
                        cnt_net_ok += 1
                    else:
                        cnt_net_ok = 0
                    if cnt_net_ok == 6:
#                         tf.train.Saver().save(sess, "./mm/mm.ckpt")
                        break                      
                   
                self.id_train.run( feed_dict = {self.x:bd_0, self.y:bd_1, self.keep_prob: self.keep_prob_rate } ) 
#             feed_content = {self.x:mnist.test.images,self.y:mnist.test.labels,self.keep_prob:1.0}
            x_test_accu_precent = self.accu_percent.eval( feed_dict={self.x: x_test, self.y:y_test, self.keep_prob:1.0} )
            print ("- the test accuracy %g " % x_test_accu_precent)
            if x_test_accu_precent > initial_accu_percent:
                tf.train.Saver().save(sess, "./mm/mm.ckpt")
                print ("- save mm.ckpt")
            
        return self        
    @property
    def flat(self):
        #self.img
        shape_line_list = self.img.shape[1:]
        type(shape_line_list)
        print (shape_line_list)
        shape_mul = 1
        for i in shape_line_list:
            #print ( int(i)+111 )
            shape_mul = shape_mul*int(i)
        shape_of_k = [-1, shape_mul ]
        img = tf.reshape( self.img, shape_of_k )
        self.img = img
        self.flat_ele = shape_mul 
        return self.img 
     

"""

(None,28*28)  : din , x 
(None, 28*28).reshape = (28,28,1) : .reshape
(28,28,1,1)*[5,5,1,32] = (28,28,32,1)  : c1
(28,28,32,1)./2 = (14,14,32,1)  : p1
(14,14,32,1)*[5,5,32,64]= (14,14,64,1)  : c2
(14,14,64,1)./2 = (7,7,64,1)  : p2
(7,7,64,1).reshape = (1,7*7*64)  : .reshape
(1,7*7*64)x[7*7*64,1024] = (1,1024) : h1
(1,1024).dropout = (1,1024)  : .dropout
(1,1024) x [1024,10] = (1,10)  : dout_ , y_

"""



if __name__ == "__main__":

    if 1:
        
        OBJ_ML_CONST={
        "lr": 2e-4,
        "iter_times": 10000,
        "batch_size": 500,
        "keep_prob_rate": 0.9 ,
        "loss": "LOSS INIT" , 
        "accu_percent": "ACCU_PERCENT INIT" , 
        "id_train": "ID_TRAIN INIT" , 
        }

#         mnist = input_data.read_data_sets( "MNIST_data/",one_hot=True)

        ops.reset_default_graph()

        x_items = 35044
        x_dim = 4
        y_dim = 3 
        
        #define data test
        x_data = y_data = x_test = y_test = np.mat("0.0;1.0")
        
        
        if 0:
            print ("- begin gen train data")
            [x_data, y_data] = gen_sets_and_label(x_items, x_dim, y_dim)
            [x_test, y_test] = gen_sets_and_label(x_items, x_dim, y_dim)
            

            np.savetxt("x_data_50000x4.nptxt",x_data, fmt="%.1f")
            np.savetxt("y_data_50000x3.nptxt",y_data, fmt="%.1f")
            np.savetxt("x_test_50000x4.nptxt",x_test, fmt="%.1f")
            np.savetxt("y_test_50000x3.nptxt",y_test, fmt="%.1f")
            
       
            
            
        if 1:
            x_data = np.loadtxt("x_data_50000x4.nptxt")
            y_data = np.loadtxt("y_data_50000x3.nptxt")            
            x_test = np.loadtxt("x_test_50000x4.nptxt")
            y_test = np.loadtxt("y_test_50000x3.nptxt")
            x_items = x_data.shape[0]
            assert(x_dim == x_data.shape[1])
            assert(y_dim == y_data.shape[1])
            
        print (x_data.shape)

        
        ml = ML( x_dim ,y_dim ,OBJ_ML_CONST )

        ml.c = ml.conv2d_by_k
        ml.p = ml.pool_2x2
        ml.cal = ml.cal_w_b_by_k
        ml.dropout=ml.tf_dropout
        

#         k1=[1,1,1,32]
#         ml.conv2d_by_k( k1, act_fun="relu" )
#         ml.pool_2x2()

#         k2=[1,1,32,1]
#         ml.conv2d_by_k(k2, act_fun="relu")
#         ml.pool_2x2()

        #print ( ml.img )
        ml.flat

        h_in_k = [ml.flat_ele, y_dim * y_dim]
        ml.cal_w_b_by_k(h_in_k, act_fun="relu")     
        
        h_in_k = [y_dim * y_dim, y_dim * y_dim ]
        ml.cal_w_b_by_k(h_in_k, act_fun="relu")
        
        h_in_k = [y_dim * y_dim , y_dim]
        ml.cal_w_b_by_k(h_in_k, act_fun="relu")
#         ml.tf_dropout( ml.keep_prob )

        h_in_k = [y_dim, y_dim]
        ml.cal_w_b_by_k(h_in_k, act_fun="softmax")        
#         ml.tf_dropout( ml.keep_prob )
        
        ml.loss_fun()
        ml.cal_accu_percent()  # self.accu_percent
        ml.set_train_args()
        

        if 0:
            print ("- start trainning process")
            ml.say()
            ml.start_train_loop(x_data, y_data, x_test,y_test)
            
        if 1:
            with tf.Session() as sess:
                
                tf.train.Saver().restore(sess, "./mm/mm.ckpt")
                
                
#                 bd_0 = x_test[0:x_items-1]
#                 bd_1 = y_test[0:x_items-1]
                
                old_accu_percent = sess.run(ml.accu_percent, feed_dict={ml.x: x_test, ml.y:y_test, ml.keep_prob: 1.0}) 
                print ("- accu_percent: %f \n" % (old_accu_percent) )
            
                y_test_ = (sess.run(ml.y_, feed_dict={ml.x:x_test, ml.keep_prob: 1.0}))
                cnt_correct = 0                
                for i, v in enumerate(y_test_):
                    if  np.argmax(v) == np.argmax(y_test[i]):
                        cnt_correct += 1
                print ("- prediction on test: %f\n\n" % (cnt_correct / len(y_test_)))
                
                
                ml.restart_train_loop(x_data, y_data, x_test,y_test, old_accu_percent)
                
                y_test_ = (sess.run(ml.y_, feed_dict={ml.x:x_test, ml.keep_prob: 1.0}))
                cnt_correct = 0   
                id_list = []
                for i, v in enumerate(y_test_):
                    if  np.argmax(v) != np.argmax(y_test[i]):
#                         print (v, " != ", y_test[i])
                        dis = np.sqrt(v[0]**2 + v[1]**2) 
#                         print (dis)
                        id_list.append(dis)
#                 id_list = [ 1,2,3,4]
                pd_t = pd.DataFrame(id_list)
                print ("- error num:", len(id_list))
#                 print(pd_t)
                pd_t.plot.bar()
               
                plt.show()
                     
                


    print ("- END")        

