import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from scipy import interpolate
from numpy.random import seed
from numpy.random import rand
from sklearn.decomposition import PCA
from tensorflow import keras
from PIL import Image
pca = PCA(n_components=1)
# seed random number generator
seed(1)
import datetime
import zipfile
import os
import toml
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
#import pywhatkit  # Import the pywhatkit library

# Define the email sending function using SendinBlue's SMTP server (without authentication)
# def send_email(recipient_email, subject, message):
#     try:
#         # Create an SMTP connection to SendinBlue's SMTP server
#         host = "server.smtp.com"
#         server = smtplib.SMTP(host)
#         FROM = "testlab@test.com"
#         TO = "ryan.parker.here@gmail.com"
#         MSG = f"Subject:{subject} \n\n{message}"
#         server.sendmail(FROM, TO, MSG)

#         server.quit()

#         st.success("Email sent successfully")
#     except Exception as e:
#         st.error(f"Email could not be sent: {str(e)}")


# Load the pickled model



def preprocessing(data):
    
  x=np.array(data[:,0])
  y=np.array(data[:,1])
  xnew=np.mean(x)+rand(5600-len(x))*np.std(x)
  xnew=np.sort(xnew)
  f = interpolate.interp1d(x, y)
  ynew=f(xnew)
  xnew=xnew.reshape(len(xnew),1)
  ynew=ynew.reshape(len(ynew),1)

  con=np.concatenate((xnew,ynew),axis=1)
  data=np.concatenate((data,con),axis=0)
  data= pd.DataFrame(data)
   
  data=data.sample(n=200,replace=True)
  data=np.array(data)
  ind = np.argsort( data[:,0] )
  data = data[ind]
  
   

  return data
import streamlit as st
import toml

# Load the secrets file
secrets = toml.load("ids.toml")

def authenticate():
    # Create a login form
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    # Check if the provided credentials match the secrets file
    if username == secrets["auth"]["username"] and password == secrets["auth"]["password"]:
        return True
    return False
        
    # Define the Streamlit app
def main():
    # Set the title of the app
    model_cd = keras.models.load_model('pickeled models\model_cd.h5')

    model_cu = keras.models.load_model('pickeled models\model_cu.h5')

    model_hg = keras.models.load_model('pickeled models\model_hg.h5')

    model_pb = keras.models.load_model('pickeled models\model_pb.h5')
    st.title('Heavy Metal Detection')
    
    if authenticate():
        img = Image.open("FLOW DIAGRAM.png")
        img1 = Image.open("mems_lab_logo.jpg")
        st.image(img, caption="Scheme For modelling used", use_column_width=True)
        st.image(img1, caption="MEMS LAB BITS Hyderabad", use_column_width=True)
        # Add a file uploader widget
        uploaded_files = st.file_uploader("Upload XLSX files", type=["xlsx"], accept_multiple_files=True)
        # If a file is uploaded
        if uploaded_files is not None:
            for uploaded_file in uploaded_files:


            
            
                
                # Read the file into a Pandas DataFrame
                df = pd.read_excel(uploaded_file)
                # Plot the data using Matplotlib
                #df=df.iloc[1600:,:]
                fig, ax = plt.subplots()
                ax.plot(df.iloc[:,0], df.iloc[:,1])
                ax.set_xlabel('voltage')
                ax.set_ylabel('current')
                ax.set_title(uploaded_file.name)
                st.pyplot(fig)
                data=preprocessing(np.array(df))
                pca.fit(data)
                data=pca.transform(data)
                data=data.reshape(200,1)
                results=""
                #Make predictions using the pickled model
                if model_cd.predict(data.reshape(1,200,1))[0]>0.5:
                   
                    results+="Cadmium is Present,   "
                if model_cu.predict(data.reshape(1,200,1))[0]>0.4:
                    
                    results+="Copper is Present,  "
                if model_hg.predict(data.reshape(1,200,1))[0]>0.5:
              
                    results+="Mercury is Present, "
                if model_pb.predict(data.reshape(1,200,1))[0]>0.5:
                   
                    results+="Lead is Present,  "
                st.write(results)
                st.write("P_cd"+str(model_cd.predict(data.reshape(1,200,1))[0]))
                st.write("P_Cu"+str(model_cu.predict(data.reshape(1,200,1))[0]))
                st.write("P_hg"+str(model_hg.predict(data.reshape(1,200,1))[0]))
                st.write("P_Pb"+str(model_pb.predict(data.reshape(1,200,1))[0]))

                # if st.button("Send Results on WhatsApp Message"):
                #     recipient_number = "+918107061807"  # Replace with the recipient's phone number
                #     results_message = results

                #     # Send the WhatsApp message using pywhatkit
                #     pywhatkit.sendwhatmsg(recipient_number, results_message, datetime.datetime.now().hour,datetime.datetime.now().minute+1)
                # Add an email button
    #     if st.button("Send Email"):
    #         recipient_email = "ryan.parker.here@gmail.com"  # Replace with the recipient's email address
    #         subject = "Heavy Metal Detection Results"
    #         results_message = results

    #         # Send the email using SendinBlue's SMTP server (no authentication)
    #         send_email(recipient_email, subject, results_message)
    # else:
    #     st.error("Authentication failed. Please enter valid credentials.")
        
if __name__ == '__main__':
    main()
