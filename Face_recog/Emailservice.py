from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
import smtplib
from email.message import EmailMessage
from email.message import EmailMessage
from email.mime.base import MIMEBase
from email import encoders

def email_alert(subject,body,to):
    msg=EmailMessage()
    msg.set_content(body)
    msg['subject']=subject 
    msg['to']=to 

    
    user ="VadimSD11@gmail.com"
    password="realpasswordforapps"
    #replace passwrod with real apppasswordforgmail
    msg['from']=user
    server= smtplib.SMTP("smtp.gmail.com",587)
    server.starttls()
    server.login(user,password)
    server.send_message(msg)
    server.quit()

def email_alert_video(subject, body, to, video_filename):
    msg = MIMEMultipart()
    msg['Subject'] = subject
    msg['From'] = "VadimSD11@gmail.com"
    msg['To'] = to

    # Attach the video file to the email
    attachment = MIMEBase('application', 'octet-stream')
    attachment.set_payload(open(video_filename, 'rb').read())
    encoders.encode_base64(attachment)
    attachment.add_header('Content-Disposition', f'attachment; filename="{video_filename}"')
    msg.attach(attachment)

    # Add body text to the email
    msg.attach(MIMEText(body, 'plain'))

    user = "VadimSD11@gmail.com"
    password = "realpasswordforapps"

    # Send the email
    with smtplib.SMTP("smtp.gmail.com", 587) as server:
        server.starttls()
        server.login(user, password)
        server.sendmail(user, to, msg.as_string())
        server.quit()