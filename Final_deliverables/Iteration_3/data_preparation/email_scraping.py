# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 06:17:43 2017

@author: anbarasan.selvarasu
"""

#Import Libraries
import urllib2 as req
from bs4 import BeautifulSoup


# Load base page

url_dir = "http://www.enron-mail.com/email/"
url_dir_open = req.urlopen(url_dir)
soup = BeautifulSoup(url_dir_open, "html.parser")
#soup.prettify()

# Get employees list to iterate

emps_list = []
mails_list = []
nav_errors_list = []
mail_errors_list = []
items = ['inbox/','sent/']
irrelevant_a = ['?C=N;O=D','?C=M;O=A','?C=S;O=A','?C=D;O=A','/']

for emp_tag in soup.select("pre a"):
    if emp_tag['href'] in irrelevant_a :
        continue
    else:
        emps_list.append(emp_tag['href'])
        
        
# List of email links  of employees inbox
"""
As the total size of this dataset is around 1.6 GB, fetching emails here for first 10 employees

"""

for emp_link in emps_list[:1]:
    for item in items:
        emp_inbox_mails_list = url_dir+ emp_link + item
        try:
            emp_inbox_mails_list_open = req.urlopen(emp_inbox_mails_list)
        except urllib.error.HTTPError as e:
            nav_errors_list.append(emp_inbox_mails_list)
            continue
        except urllib.error.URLError as e:
            nav_errors_list.append(emp_inbox_mails_list)
            continue

        emp_inbox_mails_page = BeautifulSoup(emp_inbox_mails_list_open, "html.parser" )
        for inbox_mail in emp_inbox_mails_page.select("div a"):
            mail_link = emp_inbox_mails_list + inbox_mail['href']
            try:
                emp_inbox_mail_open = req.urlopen(mail_link)
            except urllib.error.HTTPError as e:
                mail_errors_list.append(mail_link)
                continue
            except urllib.error.URLError as e:
                mail_errors_list.append(mail_link)
                continue

            emp_inbox_mails_page = BeautifulSoup(emp_inbox_mail_open, "html.parser" )
            mails_list.append(mail_link)

print(len(mails_list))

# Extracting Relevant Information

enron_emails = []

for mail_link in mails_list[:10000]:
    page = req.urlopen(mail_link)
    page_soup = BeautifulSoup(page, "html.parser")
    
    email_details = {}
    
    email_details['Title'] = page_soup.title.string
    
    
    email_header_rows = page_soup.select(".header tr")
    
    email_details['From'] = email_header_rows[0].findAll('td')[1].text.strip()
    email_details['To'] = email_header_rows[1].findAll('td')[1].text.strip()
    email_details['Subject'] = email_header_rows[2].findAll('td')[1].text.strip()
    email_details['Cc'] = email_header_rows[3].findAll('td')[1].text.strip()
    email_details['Bcc'] = email_header_rows[4].findAll('td')[1].text.strip()
    email_details['Date'] = email_header_rows[5].findAll('td')[1].text.strip()
    
    ebody_div = page_soup.find('div', {'class' :'ebody'})
    email_details['Body'] = ebody_div.text.strip('\t\r\n')
    
    enron_emails.append(email_details)

# Write the Dataframe    
import pandas as pd

email_df = pd.DataFrame(enron_emails, columns = ['From', 'To','Cc','Bcc','Subject','Body','Date','Title'])
email_df.head()

import sys
sys.setrecursionlimit(10000)

email_df.to_pickle("Enron_Email_DF.pickle") 
