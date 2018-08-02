#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 12:32:11 2018

@author: yaopeng
"""
import numpy as np

class RNNBinaryAdd:
    '''
    by default set the hidden layer size to 16
    '''
    def __init__(self, hid_dim = 16):
        
        np.random.seed(0)
        
        '''
        define the network parameters
        
        input layer dimension: 1 x N
        hidden layer dimension: 1 x H
        output layer dimension: 1 x O
        
        Then:
            input layer weight dimension: N x H
            state weight dimension: H x H
            
            the  output of the previous and current sate output dimension: 1 x H
            ( input*input_weight + pre_state*state_weight )
            hidden layer weight dimension: H x O
        
        '''
        self.input_dim = 2
        self.hid_dim = hid_dim
        self.output_dim = 1
        
        '''
        we will train the model on this binary dim.
        '''
        self.binary_dim = 8
        
        '''
        the max num of an 8 bit binary value
        '''
        self.max_num = np.power(2, self.binary_dim)
        
        '''
        define the number of state in this RNN.
        we set it the same with the binary_dim here
        '''
        self.state_num = 8
        
        '''
        np.random returns random floats in the half-open interval [0.0, 1.0)
        initialize the weights with random floats between -1 to 1
        '''
        self.weight_input = 2*np.random.random((self.input_dim, self.hid_dim)) - 1 
        self.weight_hid = 2*np.random.random((self.hid_dim, self.output_dim)) - 1 
        self.weight_state = 2*np.random.random((self.hid_dim, self.hid_dim)) - 1 
        
        '''
        save the output at each state
        '''
        self.state_vectors = np.zeros((self.state_num, self.hid_dim))
        
        '''
        the binary bit sequence of the input value
        '''
        self.X_seq = np.zeros((self.state_num, self.input_dim))
        
        
        '''
        the binary bit sequence of the target value
        '''
        self.target_seq = np.zeros((self.state_num, self.output_dim))
        
        '''
        the binary bit sequence of the predicated output value
        '''
        self.output_seq = np.zeros((self.state_num, self.output_dim))
        
        '''
        save the error in each state
        '''
        self.error_seq = np.zeros((self.state_num, self.output_dim))
        
    '''
    reset all the parameters if in order to retrain the network 
    with differnt settings: such as hidden dimension, training sample size, epoch
    '''
    def reset(self):
        self.weight_input = 2*np.random.random((self.input_dim, self.hid_dim)) - 1 
        self.weight_hid = 2*np.random.random((self.hid_dim, self.output_dim)) - 1 
        self.weight_state = 2*np.random.random((self.hid_dim, self.hid_dim)) - 1 
        
        self.state_vectors *= 0
        
        self.X_seq *= 0
        
        self.target_seq *= 0
        
        self.output_seq *= 0
        
        self.error_seq *= 0

    '''
    we use sigmod function as our activation function
    '''
    def sigmoid(self, x):
        result = 1/(1 + np.exp(-x))
        return result
        
    '''
    calculate the output at state n
    '''
    def stateFun(self, n):
        '''
        get the input at state n
        '''
        X = self.X_seq[n]
        
        '''
        get the previous state value
        '''
        pre_state = self.getState(n-1)
      
        '''
        calculate the state output according the RNN forward propagation formula
        '''
        state_vec = self.sigmoid( np.dot(X, self.weight_input) + np.dot(pre_state, self.weight_state) )
     
        '''
        save the each sate output
        '''
        self.setState(n, state_vec)
         
        return state_vec
    
    '''
    get the state vector at state n
    '''
    def getState(self, n):

        if n < 0:
            '''
            It is supposed to receive a previous state to calculate current state output,
            but at state 0, there is no previous state, so zeros are returned
            '''
            return np.zeros_like(self.state_vectors[0])
        else:
            '''
            In other cases, just return the stored state output vectors at state n
            '''
            return self.state_vectors[n]
        
    '''
    save the state output at state n
    '''
    def setState(self, n, state_vec):
        self.state_vectors[n] = state_vec
        
    '''
    calculate the final output according to the RNN forward propagation formula
    '''
    def outputFun(self, n):
        state_vec = self.stateFun(n)
        output = self.sigmoid( np.dot(state_vec, self.weight_hid) )
        return output
    
    '''
    the derivative of the sigmoid function
    '''
    def sigmoidDerivative(self, sigmoid):
        result = sigmoid * ( 1 - sigmoid )
        return result
    
    '''
    error function: MSE
    '''
    def errorFun(self, target, output):       
        result = np.power((target - output),2)/2.0
        return result
    
    '''
    the derivative of the error function respect to hidden layer output at state n
    NOTE:
        MSE function is applied here
    '''
    def errorDerivativeOutput(self, n):
        output = self.output_seq[n]
        target = self.target_seq[n]
        
        result = (target - output) * (-1)
        
        return result
    
     
    '''
    the derivative of the error with respect to the hidden layer weight at state n
    '''
    def errorDerivativeHiddenWeight(self, n):
        state_vec = self.getState(n)
        output = self.output_seq[n]
        return np.atleast_2d(state_vec).T.dot( self.errorDerivativeOutput(n)*self.sigmoidDerivative(output) )
    
    '''
    the derivative of the error with respect to the state n
    There are two back propagation path of the error:
        1) from current state output error
        2) from all the future state output error
        
    NOTE:
        At the last state, also the start point of the back propagation, there is 
        NO FUTURE state error 
    '''
    def errorDerivativeState(self, n):
        
        if n == self.state_num:
            return np.zeros_like(self.state_vectors[0])
        
        output = self.output_seq[n]
        
        '''
        path 1: error from current output
        '''
        error_current_BP = self.errorDerivativeOutput(n)*self.sigmoidDerivative(output) * np.atleast_2d(self.weight_hid).T
        
        '''
        path 2: error from all the  future 
        '''
        error_future_BP = self.errorDerivativeState(n+1)
        
        result = error_current_BP + error_future_BP
        
        return result
        
    '''
    the derivative of the error function with respect to the input weights at state n
    '''
    def errorDerivativeInputWeight(self, n):
        
        state_vec = self.getState(n)
        X = self.X_seq[n]
        
        result = np.atleast_2d(X).T.dot( state_vec * (1 -state_vec) * self.errorDerivativeState(n) )
        
        return result
    
    
    '''
    the derivative of the error function with respect to the state weights at state n
    '''
    def errorDerivativeStateWeight(self, n):
        state_vec_pre = self.getState(n-1)
        state_vec_cur = self.getState(n)
        
        result = np.atleast_2d(state_vec_pre).T.dot( state_vec_cur * (1 - state_vec_cur) * self.errorDerivativeState(n) )
        
        return result
    
    '''
    generate data samples
    '''
    def generate_training_data(self):
        #max_num = np.power(2, binary_dim)
        
        a = np.random.randint(self.max_num/2)
        b = np.random.randint(self.max_num/2)
        
        c = a + b
        
        a_binary = np.unpackbits(np.array(a, dtype=np.uint8))
        b_binary = np.unpackbits(np.array(b, dtype=np.uint8))
        
        c_binary = np.unpackbits(np.array(c, dtype=np.uint8))
    
        X = np.array([a_binary,b_binary]).T
        target = np.array([c_binary]).T
        
        return (X, target, a, b, c)
    
    '''
    A test function to run the binary add once the training is done
    '''
    def rnnBinaryAdd(self, x, y):
        
        z = x + y
        
        if z > self.max_num:
            print('the sum of the {} and {} is larger than the max 8 bit binary number 256'.format(x,y))
            return
        
        x_binary = np.unpackbits(np.array(x, dtype=np.uint8))
        y_binary = np.unpackbits(np.array(y, dtype=np.uint8))
        
        z_binary = np.unpackbits(np.array(z, dtype=np.uint8))
        
        X = np.array([x_binary,y_binary]).T
        target = np.array([z_binary]).T
        
        '''
        reverse the sequence
        '''
        X = X[::-1]
        target = target[::-1]
        
        '''
        feed to the network
        '''
        self.X_seq = X
        self.target_seq = target
        
        output=np.zeros(self.state_num)
        
        '''
        just simply go through the forward propagation
        '''
        for n in range(self.state_num):
            output[n] = self.outputFun(n)
            
        output = np.round(output)
        
        c = 0
        
        '''
        convert the binary output to decimal 
        '''
        idx = 0
        for i in output:
            c += np.power(2, idx)*i
            idx += 1
            
        print('The true answer is: {}'.format(z))
        print('The predication is: {}'.format(c))



    def train(self, alpha=0.1, epoch=1, train_sample=10000):

        '''
        define the learning rate
        '''
        self.alpha = alpha
        
        
        '''
        generate training date
        '''
        training_data = []
        
        for i in range(train_sample):
            training_data.append(self.generate_training_data())
        
        '''
        set the epoch to train the model
        '''
        self.epoch = epoch
        
        '''
        control the print 
        '''
        count = 0
        
        for e in range(self.epoch):
        
            for i in training_data:
                
                self.X_seq = i[0]
                self.target_seq = i[1]
                
                '''
                reverse the sequence as we do the math from the 
                least significant bit which is the last eliment in our list
                '''
                self.X_seq = self.X_seq[::-1]
                self.target_seq = self.target_seq[::-1]
                
                '''
                start the forward propagation to yield the output
                '''
                for n in range(self.state_num):
                    output = self.outputFun(n)
                    self.output_seq[n] = output
                
                '''
                initiate the delta weight to 0
                '''
                weight_input_delta = np.zeros_like(self.weight_input)
                weight_hid_delta = np.zeros_like(self.weight_hid)
                weight_state_delta = np.zeros_like(self.weight_state)
                
                '''
                Back Propagation Through Time (BPTT)
                As in every state, we share the weights, so in the back propagation
                we also need to add the error up in order to update each weight
                '''
                for n in range(self.state_num):
                    weight_input_delta += self.errorDerivativeInputWeight(self.state_num-n-1)
                    weight_hid_delta += np.atleast_2d(self.errorDerivativeHiddenWeight(self.state_num-n-1)).T
                    weight_state_delta += self.errorDerivativeStateWeight(self.state_num-n-1)
                    
                '''
                update each weight with the learning rate alpha
                '''
                self.weight_input -= self.alpha * weight_input_delta
                self.weight_hid -= self.alpha * weight_hid_delta
                self.weight_state -= self.alpha * weight_state_delta
                
                '''
                print the predict output and the target in binary and 
                overall error in 10 times during the training process
                '''
                if count % (len(training_data)/10) == 0:
                    '''
                    MSE
                    '''
                    '''
                    overallerror = 0
                    for i in list(zip(self.target_seq, self.output_seq)):
                        overallerror += self.errorFun(i[0], i[1])
                    '''
                    
                    '''
                    ABS error
                    '''
                    overallerror = np.sum(np.abs(self.target_seq - self.output_seq))
                    print('target     :{}'.format(list(map(int, np.round(self.target_seq)))))
                    print('predict rnd:{}'.format(list(map(int, np.round(self.output_seq)))))
                    print('predict raw:[' + ', '.join('{:0.2f}'.format(i[0]) for i in self.output_seq) + ']')
                    print('Overall abs error: {}\n'.format(overallerror))
                
                count += 1
        
        
        


