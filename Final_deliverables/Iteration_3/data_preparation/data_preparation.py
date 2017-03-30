# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 06:40:48 2017

@author: anbarasan.selvarasu
"""

from os import listdir
from os.path import isfile, join
import email
from collections import defaultdict
from utility_func.normalization import LinguisticProcessing 


class DataPreparation(object):
    
    def __init__(self,basedir):
        self.basedir = basedir
        

    def getbody(self,message): #getting plain text 'email body'
        #message = msg
        body = None
        if message.is_multipart():
            for part in message.walk():
                if part.is_multipart():
                    for subpart in part.walk():
                        if subpart.get_content_type() == 'text/plain':
                            body = subpart.get_payload(decode=True)
                elif part.get_content_type() == 'text/plain':
                    body = part.get_payload(decode=True)
        elif message.get_content_type() == 'text/plain':
            body = message.get_payload(decode=True)
        return body
    
    def get_inbox_of(self):
        #basedir = "D:/Enron/data/enron_mail_short/maildir/"
         employees = [e for e in listdir(self.basedir)]
         emp_doc = defaultdict(list) # dictionary of List
         # employees=['blair-l', 'brawner-s', 'buy-r']
         # emp = 'blair-l'
         for emp in employees:
             mail_box = [m for m in listdir(join(self.basedir,emp))]
             #print mail_box
             try:                 
                 if ('inbox' in mail_box):
                     rec = join(self.basedir,emp,"inbox").replace("\\","/")
                     msgs = [m for m in listdir(rec)]
                     file_paths = []
                     total_content = []
                     for i in msgs:
                         file_path = join(rec,i).replace("\\","/")
                         try:
                             file_handle = open(file_path,"r")
                         except: 
                             pass
                         file_content  = file_handle.read()
                         msg = email.message_from_string(file_content)
                         body = self.getbody(msg)
                         total_content.append(body)
                     emp_doc[emp].append(total_content)
                         
                 else:
                     print "Inbox not Found"
             except:
                raise ValueError(mail_box)
         return emp_doc
    
    def construct_documents(self,mailbox_dict):
        mailbox_doc = {}
        for employee,mail in list(mailbox_dict.items()):
            print(employee)
            print len(mail[0])
            try:
                objLp = LinguisticProcessing()
                list_norm_emails = objLp.normalize_corpus(mail[0],only_text_chars=True,lemmatize=True)
                mailbox_doc[employee] = ''.join(list_norm_emails)
                print list_norm_emails                
                print (mailbox_doc.keys())
                print len(mailbox_doc[employee])
            except :
                raise ValueError("Document No change Failed")
        return mailbox_doc
        
    #onlyfiles = [f for f in listdir(basdir) if isfile(join(basdir, f))]
    
    