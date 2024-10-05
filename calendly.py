import os
import requests
from dotenv import load_dotenv

load_dotenv()

def list_available_event_type_uuids():
    '''List available event type UUIDs from the Calendly account'''
    api_key = os.getenv('CALENDLY_API_KEY')
    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json'
    }
    url = 'https://api.calendly.com/event_types'
    
    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        event_types = data.get('collection', [])
        uuids = [event_type['uri'].split('/')[-1] for event_type in event_types]
        return uuids
    except requests.RequestException as e:
        return f"Failed to retrieve event types: {str(e)}"

def generate_calendly_invitation_link(query, event_type_uuid=None):
    '''Generate a calendly invitation link based on the single query string'''
    if not event_type_uuid:
        available_uuids = list_available_event_type_uuids()
        if isinstance(available_uuids, str):
            return available_uuids  # Return error message if failed to retrieve UUIDs
        elif available_uuids:
            event_type_uuid = available_uuids[0]  # Use the first available UUID
        else:
            return "No available event types found in your Calendly account."

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
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        response.raise_for_status()  # Raise an error for bad responses
        data = response.json()
        if 'resource' in data and 'booking_url' in data['resource']:
            return f"url: {data['resource']['booking_url']}"
        else:
            return "Unexpected response structure."
    except requests.RequestException as e:
        return f"Failed to create Calendly link: {str(e)}"
    
# Example usage
print(generate_calendly_invitation_link('test'))
