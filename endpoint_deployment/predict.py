import http.client
import json

# Parse the URL to connect
conn = http.client.HTTPSConnection("flask-service-681553118721.us-central1.run.app")

# Prepare the JSON payload
payload = json.dumps({
    "instances": [
        [2.3407014970657998, 2.4594060470165027, 3.024640805476224, 3.179658388901441,
         2.9151175097788653, 2.1352107522181587, 2.6082494498528423, 2.6757891667428324,
         2.400716683994664, 0.36545726606972523, 0.4205211380627995, 0.4162462629814901,
         2.5199614732626525, 2.5693165132876743, 2.555998428428869, -0.1187045499507029,
         -0.6839393084104244, 20.0, 4.0, 7.0, 1.0, -0.8660254037844386, 0.5000000000000001,
         -0.433883739117558, -0.9009688679024191]
    ]
})

# Set headers
headers = {"Content-Type": "application/json"}

# Send the POST request
conn.request("POST", "/predict", payload, headers)

# Get the response
response = conn.getresponse()
print("Status Code:", response.status)
print("Response:", response.read().decode())
