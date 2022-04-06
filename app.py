# -*- coding: utf-8 -*-
"""
Created on Thu Feb 10 10:41:19 2022

@author: User
"""

from flask import Flask, render_template, url_for, request
import pandas as pd 
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.externals import joblib

from flask import Flask
app=Flask(__name__)
from flask import request,render_template


    
    
@app.route("/",methods=["GET","POST"])
def index():
    if request.method=="POST":
        description=request.form.get("description")
        NBmodel = open('NBmodel.pkl','rb') #open in read binary
        clf = joblib.load(NBmodel)
        cv_model = open('cv.pkl', 'rb')
        cv = joblib.load(cv_model)
        data = [description]
        vect = cv.transform(data).toarray()
        pred = clf.predict(vect)        
        print(pred)
        s="The predicted theme is " + str(pred)
        return(render_template("index.html",result=s))
    else:
        return(render_template("index.html",result="Predict 2"))
    
if __name__=="__main__":
    app.run()
    
    


# @app.route("/",methods=["GET","POST"])
# def index():
#     if request.method=="POST":
#         description=request.form.get("description")
#         model=joblib.load("Q1")
#         pred=model.predict(description)
#         print(pred)
#         pred=pred[0]
#         s="The predicted theme is " + str(pred)
#         return(render_template("index.html",result=s))
#     else:
#         return(render_template("index.html",result="Predict 2"))
    
# if __name__=="__main__":
#     app.run()
