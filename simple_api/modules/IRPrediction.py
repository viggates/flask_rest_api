# -*- coding: utf-8 -*-
import nltk
import mysql.connector
import sys
import datetime
import re
import timeit
from nltk.corpus import stopwords
from nltk.stem.snowball import SnowballStemmer
from sklearn.preprocessing import LabelBinarizer
import pandas as pd
import matplotlib.pyplot as plt
from textblob import TextBlob
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier 
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
#from imblearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split , StratifiedShuffleSplit,learning_curve,GridSearchCV
import logging
from multiprocessing import Pool
import pickle


class Prediction(object):

    pkl_file = "/etc/innobotz/decison_tree.pkl"
    model_reload = pickle.load(open(pkl_file, mode='rb'))
    #predict='assignee_grp'
    num_partitions = 4
    num_cores = 4
    loop = 0

    mysw = []
    stem = []

    def __init__(self, ir_no):
        self.ir_no = ir_no
        print(ir_no)
        mysw = []
        for sw in stopwords.words('english'):
            mysw = mysw + [sw]
        mysw = mysw + list(['monday','tuesday','wednesday','thursday','friday'])
        mysw=mysw+ list(['mon','tue','wed','thu','fri','saturday','sunday','sat','sun'])
        mysw = mysw + list(['description','custom','impact','ct','summary','ciname',
                       'report','client','process','monitor','event','server'])
        mysw = mysw + list(['ticket','target','file','central','completed','complete',
                        'notify','please','assign','ir','product','originating',
                        'support','ca7'])
        mysw = mysw + list(['current','number','job','data','system','team','application'])
        mysw = mysw + list(['january','february','march','april','may','june','july','august',
                        'september','october','november','december','jan','feb','mar',
                        'apr','may','jun','jul','aug','sep','oct','nov','dec','day',
                        'date','month','mon','seconds'])
        #more stop words
        for sw in ('in','at','please','null','nvl','name','user','line','queue','0','000','00','0000','00000','000000','0000000','00000000'):
            mysw = mysw + [sw]
        for i in range(100,9999):
            mysw = mysw + [str(i)]
        self.mysw = mysw

    def get_mysql_conn(self):
        conn = mysql.connector.connect(user='sysadm', password='Car3fu!!', host='sdnew.cmkw1wmgnizt.us-east-1.rds.amazonaws.com', database='servicedesk')
        return conn

    def parallelize_dataframe(self, df, func):
        df_split = np.array_split(df, self.num_partitions)
        pool = Pool(self.num_cores)
        df = pd.concat(pool.map(func, df_split))
        pool.close()
        pool.join()
        return df

    def split_into_lemmas(self, message):
        global removed
        global stem ,stem1
        #print(message)
        message = str(message).lower()
        #message = re.sub(r'(?:(?:(?:(\d+):)?(\d+):)?(\d+)(am|ct|cst|pm|czt|cdt)?)','',message)
        #message = re.sub(r'[a-zA-Z0-9]*(\d+)_oma[a-zA-Z0-9_\-]*','oma',message)
        #message = re.sub(r'[a-zA-Z0-9]*(\d+)_hag[a-zA-Z0-9_\-]*','hag',message)
        #message = re.sub(r'(£)?\b(\d+)[,.](\d+).(\d+)\b','',message)
        stemmer =SnowballStemmer('english')
        #print(message)
        words=TextBlob(message).words
        new_word=''
        stem1=[]
        if self.loop%1000 ==0:
            print(self.loop)
        self.loop = self.loop + 1
        for word in words:
            word=word.replace('"','',3)
            word=word.replace('“','',3)
            word=word.replace('”','',3)
            word=word.replace(',','',3)
            word=word.replace("'",'',3)
            if word.isnumeric() or (word in self.mysw) or re.search('^(?:(?:(?:(\d+)[:/.-])?(\d+)[:/.-])?(\d+)(am|ct|cst|pm|czt|cdt)?)$',word) or re.search('^(\d){2}[:](\d){2}[:](\d){2}[.:](\d+)$',word) or re.search('^(\d+)[.]?(\d+)?$',word) or re.search('^(tokenid=|alarmid:)(.)*$',word) or re.search('^j(\d+)$',word):
                #return 'bubadubadeeba'
    #            if word not in removed:
    #               removed = removed + [word]
                continue
            else:
                stem_word= stemmer.stem(word)
    #            if stem_word not in stem:
    #                if stem_word not in mysw:
    #                    stem =stem + [stem_word]
                if stem_word not in stem1:
                    stem1= stem1 + [stem_word]
                    new_word= new_word + stem_word + ' '
    #    if loop == 80457:
    #        pd.DataFrame(stem).to_excel("Stem.xlsx")
        return new_word

    def parallel_split(self, data):
        data['nap'] = data['append'].map(lambda text:self.split_into_lemmas(text))
        return data

    def resolve(self, predict):
        if predict=='ResolutionCodeDesc':
            df3 = pd.read_excel('/home/vm-user/acoe/IR_RESCD_P12345_2017.xlsx','Sheet1')
        if predict=='assignee_grp':
            conn = self.get_mysql_conn()
            #df3 = pd.read_excel('/home/vm-user/acoe/IR_ASGGRP_P12345_2017.xlsx','Sheet1')
            #df3 = pd.read_sql('SELECT * FROM incident_data limit 1', con=conn)
            #df3 = pd.read_sql('SELECT * FROM incident_data where incident_no=%d' %2342, con=conn)
            df3 = pd.read_sql('SELECT * FROM incident_data where incident_no="{0}"'.format(self.ir_no), con=conn)
            if df3.empty:
                return ["None"]
            #df3 = pd.read_sql('SELECT * FROM incident_data ', con=conn)
        df4 = df3
        #df4=df3[:][['incident_no', 'incident_area', 'incident_lob', 'reportedbygroup', 'assignee_grp', 'assignee', 'resolving_grp', 'priority', 'severity', 'urgency', 'summary', 'description', 'configuration_item', 'incident_status', 'resolution_code', 'major_incident', 'reporting_method', 'open_datetime', 'resolution_method', 'tech_descr', 'tech_resolution', 'region']]
        #df4 = df4.dropna(subset=[predict])
        #df4 = df4.reset_index()
        #Strip patterns
        print('before split',timeit.default_timer())
        df4['incident_area']=df4['incident_area'].fillna(" ")
        #df4['ResolvedByGroupName']=df4['ResolvedByGroupName'].fillna(" ")
        df4['summary']=df4['summary'].fillna(" ")
        df4['incident_lob']=df4['incident_lob'].fillna(" ")
        df4['reportedbygroup']=df4['reportedbygroup'].fillna(" ")
        df4['assignee_grp']=df4['assignee_grp'].fillna(" ")
        df4['assignee']=df4['assignee'].fillna(" ")
        df4['resolving_grp']=df4['resolving_grp'].fillna(" ")
        df4['priority']=df4['priority'].fillna(" ")
        df4['severity']=df4['severity'].fillna(" ")
        df4['urgency']=df4['urgency'].fillna(" ")
        df4['summary']=df4['summary'].fillna(" ")
        df4['description']=df4['description'].fillna(" ")
        df4['configuration_item']=df4['configuration_item'].fillna(" ")
        df4['incident_status']=df4['incident_status'].fillna(" ")
        df4['resolution_code']=df4['resolution_code'].fillna(" ")
        df4['major_incident']=df4['major_incident'].fillna(" ")
        df4['reporting_method']=df4['reporting_method'].fillna(" ")
        df4['resolution_method']=df4['resolution_method'].fillna(" ")
        df4['tech_descr']=df4['tech_descr'].fillna(" ")
        df4['tech_resolution']=df4['tech_resolution'].fillna(" ")
        df4['region']=df4['region'].fillna(" ")
        df4['resolution1'],df4['resolution2']=df4['resolution_code'].str.split('.').str
        df4['formatstr'] = df4['incident_area'].map(lambda text:text.lower())
        df4['append'] = df4['summary'].astype(str)
        df4=df4.drop('summary',axis=1)
        df4['formatstr'] = df4['formatstr'] + ' '

        df5 = df4.groupby([predict])[predict].count()
        df7 = df4.groupby([predict])[predict].count()
        df4[-df4[predict].isin(list(df7.index.get_level_values(0)))]='Other'
        df6=pd.DataFrame({predict:df5.index,'resolution_code':list(range(df5.count())),'Count':df5.values})
        df_temp= df4[['append',predict,'resolution1','resolution2']].to_excel("IRP12345NEW2017_text.xlsx",index=False)

        #df_temp= df4[[predict,'resolution1','resolution2','Newdescr']].to_excel("IRP12345NEW2017_text_lemma.xlsx",index=False)
        new_data=df4[['append','formatstr',predict,'resolution1','resolution2']]
        to_predict=new_data[:]
        to_predict=pd.merge(to_predict,df6,on=predict)
        to_predict= to_predict.reset_index()
        #import pdb;pdb.set_trace()
        #to_predict= self.parallelize_dataframe(to_predict,parallel_split)
        to_predict= self.parallel_split(to_predict)
        print(to_predict)
        to_predict['formatstr']=to_predict['formatstr'] + ' ' +  to_predict['nap']
        df_temp= to_predict[['formatstr',predict,'resolution1','resolution2']].to_excel("IRP12345TEST2017_text_lemma.xlsx",index=False)
        print(to_predict['formatstr'])
        results=self.model_reload.predict(to_predict['formatstr'])
        print(results)
        return results

if __name__ == "__main__":
    import pdb;pdb.set_trace()
    px = Prediction()
    px.resolve("assignee_grp")
