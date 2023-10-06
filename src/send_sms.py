import smtplib
import time
from email.mime.text import MIMEText

# Email configuration
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587
SENDER_EMAIL = 'klaudia1232sms@gmail.com'
SENDER_PASSWORD = 'smseric123'
RECIPIENT_PHONE = '0048696894414'  # Replace with recipient's phone number
RECIPIENT_CARRIER_GATEWAY = 'recipient_carrier_gateway'  # Replace with recipient's carrier gateway address

def send_email_as_text(subject, message):
    recipient_email = f"{RECIPIENT_PHONE}@{RECIPIENT_CARRIER_GATEWAY}"
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = recipient_email

    try:
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SENDER_EMAIL, SENDER_PASSWORD)
        server.sendmail(SENDER_EMAIL, recipient_email, msg.as_string())
        server.quit()
        print("Text message sent successfully.")
    except Exception as e:
        print("Error sending text message:", str(e))

def your_script():
    # Simulating a long-running script
    for i in range(5):
        print(f"Processing step {i+1}")
        time.sleep(2)

if __name__ == "__main__":
    try:
        start_time = time.time()
        your_script()
        end_time = time.time()
        execution_time = end_time - start_time
        message = f"Script execution finished. Execution time: {execution_time:.2f} seconds."
        send_email_as_text("Script Execution Finished", message)
    except Exception as e:
        error_message = f"Script execution failed: {str(e)}"
        send_email_as_text("Script Execution Failed", error_message)
