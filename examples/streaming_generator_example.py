import os
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://api-inference.huggingface.co/models/facebook/bart-large"
headers = {"Authorization": f"Bearer {os.getenv('HUGGING_FACE_API_KEY')}"}

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

# Set up your agent specifics
salesperson_name = "Ted Lasso"
salesperson_role = "Sales Representative"
company_name = "Sleep Haven"
company_business = """Sleep Haven 
                            is a premium mattress company that provides
                            customers with the most comfortable and
                            supportive sleeping experience possible. 
                            We offer a range of high-quality mattresses,
                            pillows, and bedding accessories 
                            that are designed to meet the unique 
                            needs of our customers."""

# Example of how to query the model
output = query({
    "inputs": "Provide a sales pitch for Sleep Haven."
})

print(output)
