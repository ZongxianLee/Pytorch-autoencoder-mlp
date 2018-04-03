
# coding: utf-8

# In[1]:


import cPickle as pickle


# In[39]:


def Get_data(file_name):
    # the train/val/test will save in a list, and the 0th item is the data and 
    # 1st item refers to its corrsponding label
    train = []
    val = []
    test = []
    fr = open(file_name)
    inf = pickle.load(fr)
    train.append(inf[0][0])
    #print(len(train[0]))
    train.append(inf[0][1])
    val.append(inf[1][0])
    val.append(inf[1][1])
    test.append(inf[2][0])
    test.append(inf[2][1])
    #print('aaa')
    return train, val, test

