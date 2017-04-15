#!/usr/bin/env python 

from util import * 
from numpy import *
from math import log
import copy
import sys


# Pretty printing for 1D/2D numpy arrays
MAX_PRINTING_SIZE = 30

def format_array(arr):
    s = shape(arr)
    if s[0] > MAX_PRINTING_SIZE or (len(s) == 2 and s[1] > MAX_PRINTING_SIZE):
        return "[  too many values (%s)   ]" % s

    if len(s) == 1:
        return  "[  " + (
            " ".join(["%.6f" % float(arr[i]) for i in range(s[0])])) + "  ]"
    else:
        lines = []
        for i in range(s[0]):
            lines.append("[  " + "  ".join(["%.6f" % float(arr[i,j]) for j in range(s[1])]) + "  ]")
        return "\n".join(lines)



def format_array_print(arr):
    print format_array(arr)


def string_of_model(model, label):
    (initial, tran_model, obs_model) = model
    return """
Model: %s 
initial: 
%s

transition: 
%s

observation: 
%s
""" % (label, 
       format_array(initial),
       format_array(tran_model),
       format_array(obs_model))

    
def check_model(model):
    """Check that things add to one as they should"""
    (initial, tran_model, obs_model) = model
    for state in range(len(initial)):
        assert((abs(sum(tran_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(obs_model[state,:]) - 1)) <= 0.01)
        assert((abs(sum(initial) - 1)) <= 0.01)


def print_model(model, label):
    check_model(model)
    print string_of_model(model, label)    

def max_delta(model, new_model):
    """Return the largest difference between any two corresponding 
    values in the models"""
    return max( [(abs(model[i] - new_model[i])).max() for i in range(len(model))] )


class HMM:
    """ HMM Class that defines the parameters for HMM """
    def __init__(self, states, outputs):
        """If the hmm is going to be trained from data with labeled states,
        states should be a list of the state names.  If the HMM is
        going to trained using EM, states can just be range(num_states)."""
        self.states = states
        self.outputs = outputs
        n_s = len(states)
        n_o = len(outputs)
        self.num_states = n_s
        self.num_outputs = n_o
        self.initial = zeros(n_s)
        self.transition = zeros([n_s,n_s])
        self.observation = zeros([n_s, n_o])

    def set_hidden_model(self, init, trans, observ):
        """ Debugging function: set the model parameters explicitly """
        self.num_states = len(init)
        self.num_outputs = len(observ[0])
        self.initial = array(init)
        self.transition = array(trans)
        self.observation = array(observ)
        
    def get_model(self):
        return (self.initial, self.transition, self.observation)

    def compute_logs(self):
        """Compute and store the logs of the model (helper)"""
        raise Exception("Not implemented")

    def __repr__(self):
        return """states = %s
observations = %s
%s
""" % (" ".join(array_to_string(self.states)), 
       " ".join(array_to_string(self.outputs)), 
       string_of_model((self.initial, self.transition, self.observation), ""))

     
    # declare the @ decorator just before the function, invokes print_timing()
    @print_timing
    def learn_from_labeled_data(self, state_seqs, obs_seqs):
        """
        Learn the parameters given state and observations sequences. 
        The ordering of states in states[i][j] must correspond with observations[i][j].
        Use Laplacian smoothing to avoid zero probabilities.
        Implement for (a).
        """

        states = unique(flatten(state_seqs))
        outputs = unique(flatten(obs_seqs))
 
        n_s = self.num_states
        n_o = self.num_outputs 

        initial = self.initial 
        transition = self.transition 
        observation = self.observation 

        #dictionary that assigns state to index, and output to index
        state_dictionary = dict(zip(states,range(len(states))))
        self.state_dictionary = state_dictionary
        output_dictionary = dict(zip(outputs,range(len(outputs))))
        self.output_dictionary = output_dictionary 
        #want number of times in each state k
        N = len(state_seqs)
        N_k = zeros(n_s) 
        N_minusnk = zeros(n_s)
        N_kl = zeros([n_s,n_s])
        N_kj = zeros([n_s,n_o])

        initial = initial + 1.0

        #loop through state sequences 
        for i in xrange(len(state_seqs)):

            #get observation sequence and state sequence
            state_seq = state_seqs[i]
            obs_seq = obs_seqs[i]

            #get index of first state in dictionary and update initial
            ind0 = state_dictionary[state_seq[0]]
            initial[ind0] += 1

            #loop through the state sequence
            for j in xrange(len(state_seq)):
                ind_j = state_dictionary[state_seq[j]]

                N_k[ind_j] += 1 

                if (j!= len(state_seq) - 1):

                    N_minusnk[ind_j] += 1

                    ind_next = state_dictionary[state_seq[j+1]]

                    N_kl[ind_j,ind_next] += 1 

                #update pi matrix
                ind_obs = output_dictionary[obs_seq[j]]
                N_kj[ind_j,ind_obs] += 1


        
        #normalize initial states, compute estimates for theta
        initial = initial / float(N+n_s)
        self.initial = initial 

        #compute estimates for transition matrix and output matrix
        # smoothing step
        N_kl = N_kl + 1.0
        N_kj = N_kj + 1.0
        N_minusnk = N_minusnk + n_s
        N_k = N_k + n_o

        #division step
        transition = divide(N_kl.T,N_minusnk).T
        observation = divide(N_kj.T,N_k).T
        self.transition = transition
        self.observation = observation

        

    def most_likely_states(self, sequence, debug=False):
        """Return the most like sequence of states given an output sequence.
        Uses Viterbi algorithm to compute this.
        Implement for (b) and (c).
        """
       
        #states = unique(flatten(state_seqs))
        #outputs = unique(flatten(obs_seqs))
 
        n_s = self.num_states
        n_o = self.num_outputs 



        #dictionary that assigns state to index, and output to index
        #state_dictionary = dict(zip(states,range(len(states))))
        #output_dictionary = dict(zip(outputs,range(len(outputs))))


        T = len(sequence)
        T1 = zeros([n_s,T])
        T2 = zeros([n_s,T])

        y1_ind = sequence[0]
        #y1_ind = output_dictionary[y1]

        for i in xrange(len(self.states)):
            T1[i,0] = log(self.initial[i]) + log(self.observation[i,y1_ind])
            T2[i,0] = 0 

        for i in xrange(1,T):
            y_i_ind = sequence[i]
            #y_i_ind = output_dictionary[y_i]

            for j in xrange(len(self.states)):

                T1kA = []
                for k in xrange(len(self.states)):
                    T1kA.append(T1[k,i-1] + log(self.transition[k,j]))
                    

                T1kA = array(T1kA)
                mx = max(T1kA)
                argmx = argmax(T1kA)

                T1[j,i] = log(self.observation[j,y_i_ind]) + mx
                T2[j,i] = argmx 


        hidden_seq_inds = zeros(T)
        hidden_seq = zeros(T)

        hidden_seq_inds[T-1] = argmax(T1[:,T-1])
        #hidden_seq[T-1] = state_dictionary[hidden_seq_inds[T-1]]
        for i in range(1,T)[::-1]:
            hidden_seq_inds[i-1] = T2[hidden_seq_inds[i].astype(int),i]
            #hidden_seq[i] = state_dictionary[hidden_seq_inds[i-1]]

        return list(hidden_seq_inds.astype(int))


    
def get_wikipedia_model():
    # From the rainy/sunny example on wikipedia (viterbi page)
    hmm = HMM(['Rainy','Sunny'], ['walk','shop','clean'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.4,0.5], [0.6,0.3,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm

def get_toy_model():
    hmm = HMM(['h1','h2'], ['A','B'])
    init = [0.6, 0.4]
    trans = [[0.7,0.3], [0.4,0.6]]
    observ = [[0.1,0.9], [0.9,0.1]]
    hmm.set_hidden_model(init, trans, observ)
    return hmm
    

