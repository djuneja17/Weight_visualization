#
# Cloud for ML Final Project
# Cole Smith
# client.py
#

import base64
import sys
from io import BytesIO

import requests
from PIL import Image

# Examples of input
# url = "http://192.168.99.102:32149"
# filepath = "n09835506_31225.JPEG"
# output path = "."

url = sys.argv[1]
filepath = sys.argv[2]

output_path = "./response.jpg"
if len(sys.argv) == 4:
    output_path = sys.argv[3]

# Open the image file for which to send
with open(filepath, "rb") as fp:
    # Read it in BYTES, encode those bytes in base64
    encoded_img = base64.b64encode(fp.read())

    # Package this into a JSON object, decode base64 bytes to UTF-8
    data = {'img': encoded_img.decode("utf-8")}

    # Send the POST request with JSON body to the server URL
    # (provided from STDIN)
    r = requests.post(url=url, json=data)

    # Print the JSON keys we got back
    print(r.json().keys())

    # Convert the blurred image base64 string to a RAW Python string
    # raw = "%r" % r.json()['blurredImage']
    raw = "%r" % r.json()['cropped']
    # raw = "%r" % r.json()['gradCAM']

    # Decode that raw base64 string to a BytesIO object
    b = BytesIO(base64.b64decode(raw))

    # Use PIL to open those image bytes
    im = Image.open(b)

    # Save that image to the output path (default response.jpg, or given by STDIN)
    im.save(output_path)
