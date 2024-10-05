import os
import logging
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def validate_env_vars():
    """Validate necessary environment variables are set."""
    required_vars = ["GMAIL_MAIL", "GMAIL_APP_PASSWORD"]
    for var in required_vars:
        if os.getenv(var) is None:
            logger.error(f"{var} not set in environment variables.")
            return False
    return True

def send_simple_email(recipient_email: str, subject: str, body: str, content_type: str = 'plain') -> str:
    """Send a simple email using Gmail SMTP.

    Args:
        recipient_email (str): The recipient's email address.
        subject (str): The subject of the email.
        body (str): The body of the email.
        content_type (str): The type of content (plain or html). Default is 'plain'.

    Returns:
        str: Result message indicating success or failure.
    """
    if not validate_env_vars():
        return "Environment variables are not properly configured."

    try:
        sender_email = os.getenv("GMAIL_MAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")

        # Create MIME message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, content_type))

        # Create server object with SSL option
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, app_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())

        logger.info("Email sent successfully to %s", recipient_email)
        return "Email sent successfully."
    
    except smtplib.SMTPException as smtp_err:
        logger.error("SMTP error occurred: %s", smtp_err)
        return f"Failed to send email due to SMTP error: {smtp_err}"
    
    except Exception as e:
        logger.error("An error occurred: %s", e)
        return f"Failed to send email: {e}"

# Test email details
recipient_email = os.getenv("GMAIL_MAIL")  # Use the email from environment variables
subject = "Test Email"
body = "This is a test email sent from the Python script without using LLM."

# Send the test email
result = send_simple_email(recipient_email, subject, body)
print(result)
