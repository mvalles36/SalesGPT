import json
import os
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
headers = {"Authorization": f"Bearer {os.getenv('HUGGGINGFACE_API_KEY')}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def setup_knowledge_base(product_catalog: str = None):
    with open(product_catalog, "r") as f:
        product_catalog_content = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    texts = text_splitter.split_text(product_catalog_content)

    knowledge_base = []  # Placeholder for knowledge base setup

    return knowledge_base


def get_product_id_from_query(query, product_price_id_mapping_path):
    with open(product_price_id_mapping_path, "r") as f:
        product_price_id_mapping = json.load(f)

    product_price_id_mapping_json_str = json.dumps(product_price_id_mapping)

    enum_list = list(product_price_id_mapping.values()) + [
        "No relevant product id found"
    ]
    enum_list_str = json.dumps(enum_list)

    prompt = f"""
    You are an expert data scientist. Given the query: {query} and the product price id mapping: {product_price_id_mapping_json_str}, return the relevant price id.
    Return a valid directly parsable json, don't return in it within a code snippet or add any kind of explanation!!
    """

    output = query({"inputs": prompt})
    product_id = output.get("generated_text", "").strip()

    return product_id


def generate_stripe_payment_link(query: str) -> str:
    PAYMENT_GATEWAY_URL = os.getenv(
        "PAYMENT_GATEWAY_URL", "https://agent-payments-gateway.vercel.app/payment"
    )
    PRODUCT_PRICE_MAPPING = os.getenv(
        "PRODUCT_PRICE_MAPPING", "example_product_price_id_mapping.json"
    )

    price_id = get_product_id_from_query(query, PRODUCT_PRICE_MAPPING)
    payload = json.dumps(
        {"prompt": query, "price_id": price_id, "stripe_key": os.getenv("STRIPE_API_KEY")}
    )
    headers = {"Content-Type": "application/json"}
    response = requests.post(PAYMENT_GATEWAY_URL, headers=headers, data=payload)
    return response.text


def get_mail_body_subject_from_query(query):
    prompt = f"""
    Given the query: "{query}", analyze and extract the necessary email information.
    Return a valid directly parsable json, don't return in it within a code snippet or add any kind of explanation!!
    """
    output = query({"inputs": prompt})
    mail_body_subject = output.get("generated_text", "").strip()
    return mail_body_subject


def send_email_with_gmail(email_details):
    try:
        sender_email = os.getenv("GMAIL_MAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        recipient_email = email_details["recipient"]
        subject = email_details["subject"]
        body = email_details["body"]
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, app_password)
        text = msg.as_string()
        server.sendmail(sender_email, recipient_email, text)
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        return f"Email was not sent successfully, error: {e}"


def send_email_tool(query):
    email_details = get_mail_body_subject_from_query(query)
    if isinstance(email_details, str):
        email_details = json.loads(email_details)
    result = send_email_with_gmail(email_details)
    return result


def generate_calendly_invitation_link(query):
    event_type_uuid = os.getenv("CALENDLY_EVENT_UUID")
    api_key = os.getenv('CALENDLY_API_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    url = 'https://api.calendly.com/scheduling_links'
    payload = {
        "max_event_count": 1,
        "owner": f"https://api.calendly.com/event_types/{event_type_uuid}",
        "owner_type": "EventType"
    }

    response = requests.post(url, json=payload, headers=headers)
    if response.status_code == 201:
        data = response.json()
        return f"url: {data['resource']['booking_url']}"
    else:
        return "Failed to create Calendly link."


def get_tools(product_catalog):
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="Useful for answering questions about product information or services.",
        ),
        Tool(
            name="GeneratePaymentLink",
            func=generate_stripe_payment_link,
            description="Generates a payment link based on the customer query.",
        ),
        Tool(
            name="SendEmail",
            func=send_email_tool,
            description="Sends an email based on the query input.",
        ),
        Tool(
            name="SendCalendlyInvitation",
            func=generate_calendly_invitation_link,
            description="Creates a Calendly invitation based on the query input.",
        )
    ]
    return tools
