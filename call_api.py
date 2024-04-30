import requests

def call_api(query, api_key):
    """Send a POST request to the API with the given query and api_key."""
    url = 'http://localhost:8000/process_query/'
    headers = {'Content-Type': 'application/json'}
    data = {'query': query, 'api_key': api_key}
    
    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()  # Raises HTTPError for bad responses (4XX, 5XX)
        return response.json()
    except requests.exceptions.HTTPError as e:
        return {'error': 'HTTP error occurred: {}'.format(e)}
    except requests.exceptions.ConnectionError:
        return {'error': 'Connection error occurred'}
    except requests.exceptions.Timeout:
        return {'error': 'Timeout error occurred'}
    except requests.exceptions.RequestException as e:
        return {'error': 'An unexpected error occurred: {}'.format(e)}
    
# # Example usage
# result = call_api("What is a VPN?", "your-api-key")
# print(result)
