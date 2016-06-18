
# coding: utf-8

# In[82]:

# 
import re;
with open('mnist2rec.md', 'r') as f:
    s = f.readlines();
    for i in range(len(s)):
        print line
        m = re.match('\!\[png\]', line)
        if m is not None:
            full = line.find('https://') + line.find('http://')
            if full < 0:
                line = list(line);
                loc = line.index('(') + 1;
                line.insert(loc, 'https://raw.githubusercontent.com/dengdan/deep-learning-with-mxnet/master/io/')
                line = ''.join(line)
                s[i] = line
with open('mnist2rec.md', 'w') as f:                
    f.writelines(s);


# In[85]:

import os;
os.system('jupyter nbconver')


# In[46]:

https://raw.githubusercontent.com/dengdan/deep-learning-with-mxnet/master/io/


# In[89]:

'asss'.endswith('a')


# In[91]:

os.path.join('ss','s')


# In[97]:

s = 's/fff';
s[s.find('/')+1: len(s)]


# In[ ]:



