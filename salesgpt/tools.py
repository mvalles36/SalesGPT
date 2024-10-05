import json
import os
import requests
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from langchain.agents import Tool
from langchain.chains import RetrievalQA
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma

# Initialize the Hugging Face API for text generation
def generate_completion(prompt, model_name="gpt2"):
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_KEY')}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "options": {"use_cache": False},
    }
    response = requests.post(f"https://api-inference.huggingface.co/models/{model_name}", headers=headers, json=payload)
    
    if response.status_code == 200:
        return response.json()[0]['generated_text']
    else:
        raise Exception(f"Error from Hugging Face API: {response.text}")

def setup_knowledge_base(product_catalog: str = None):
    """
    Load product catalog and create a knowledge base.
    """
    # Load product catalog
    with open(product_catalog, "r") as f:
        product_catalog_text = f.read()

    text_splitter = CharacterTextSplitter(chunk_size=5000, chunk_overlap=200)
    texts = text_splitter.split_text(product_catalog_text)

    # Use a vector store for document retrieval
    docsearch = Chroma.from_texts(texts, collection_name="product-knowledge-base")

    knowledge_base = RetrievalQA.from_chain_type(
        llm=None,  # No LLM needed for this part
        chain_type="stuff",
        retriever=docsearch.as_retriever()
    )
    return knowledge_base

def get_product_id_from_query(query, product_price_id_mapping_path):
    # Load product_price_id_mapping from a JSON file
    with open(product_price_id_mapping_path, "r") as f:
        product_price_id_mapping = json.load(f)

    # Create a prompt for the Hugging Face model
    product_price_id_mapping_json_str = json.dumps(product_price_id_mapping)
    enum_list = list(product_price_id_mapping.values()) + ["No relevant product id found"]
    enum_list_str = json.dumps(enum_list)

    prompt = f"""
    You are an expert data scientist working to recommend products based on customer queries.
    Given the query: {query}
    and the product price id mapping: {product_price_id_mapping_json_str},
    return the most relevant price id. ONLY return the price id.
    Your output must be valid JSON.
    {{
        "$schema": "http://json-schema.org/draft-07/schema#",
        "title": "Price ID Response",
        "type": "object",
        "properties": {{
            "price_id": {{
                "type": "string",
                "enum": {enum_list_str}
            }}
        }},
        "required": ["price_id"]
    }}
    """
    
    response = generate_completion(prompt, model_name="gpt2")
    product_id = json.loads(response)
    return product_id

def generate_stripe_payment_link(query: str) -> str:
    """Generate a stripe payment link for a customer based on a single query string."""
    PAYMENT_GATEWAY_URL = os.getenv("PAYMENT_GATEWAY_URL", "https://agent-payments-gateway.vercel.app/payment")
    PRODUCT_PRICE_MAPPING = os.getenv("PRODUCT_PRICE_MAPPING", "example_product_price_id_mapping.json")

    # Use LLM to get the price_id from the query
    price_id = get_product_id_from_query(query, PRODUCT_PRICE_MAPPING)
    payload = json.dumps({"prompt": query, **price_id, "stripe_key": os.getenv("STRIPE_API_KEY")})
    headers = {"Content-Type": "application/json"}

    response = requests.post(PAYMENT_GATEWAY_URL, headers=headers, data=payload)
    return response.text

def get_mail_body_subject_from_query(query):
    prompt = f"""
    Given the query: "{query}", extract the recipient's email, subject, and body for sending an email.
    Return a dictionary in JSON format with keys 'recipient', 'subject', and 'body'.
    """
    
    response = generate_completion(prompt, model_name="gpt2")
    mail_body_subject = json.loads(response)
    return mail_body_subject

def send_email_with_gmail(email_details):
    """Send an email using Gmail."""
    try:
        sender_email = os.getenv("GMAIL_MAIL")
        app_password = os.getenv("GMAIL_APP_PASSWORD")
        recipient_email = email_details["recipient"]
        subject = email_details["subject"]
        body = email_details["body"]

        # Create MIME message
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = recipient_email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))

        # Create server object with SSL option
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465)
        server.login(sender_email, app_password)
        server.sendmail(sender_email, recipient_email, msg.as_string())
        server.quit()
        return "Email sent successfully."
    except Exception as e:
        return f"Email was not sent successfully, error: {e}"

def send_email_tool(query):
    """Sends an email based on the query string."""
    email_details = get_mail_body_subject_from_query(query)
    result = send_email_with_gmail(email_details)
    return result

def generate_calendly_invitation_link(query):
    """Generate a Calendly invitation link based on a single query string."""
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
    """Setup tools for the agent."""
    knowledge_base = setup_knowledge_base(product_catalog)
    tools = [
        Tool(
            name="ProductSearch",
            func=knowledge_base.run,
            description="Useful for answering questions about products and their availability.",
        ),
        Tool(
            name="GeneratePaymentLink",
            func=generate_stripe_payment_link,
            description="Generates a payment link based on the product and customer details.",
        ),
        Tool(
            name="SendEmail",
            func=send_email_tool,
            description="Sends an email based on the query input.",
        ),
        Tool(
            name="SendCalendlyInvitation",
            func=generate_calendly_invitation_link,
            description="Creates a Calendly invitation link based on the input query.",
        )
    ]

    return tools
