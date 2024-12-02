import numpy as np

nums=np.random.rand(30)

#print(nums)

from joblib import load

model_rf=load('rf.joblib')


from flask import Flask
app=Flask(__name__)

@app.route('/')
def hello ():
    return 'Hello world'
#new line modified (added after push and pull)