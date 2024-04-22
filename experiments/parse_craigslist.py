import logging
import os
import time

import requests
from bs4 import BeautifulSoup

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# URL of the Craigslist search results page
url = "https://poconos.craigslist.org/search/apa?hasPic=1#search=1~gallery~0~0"

# Send a GET request to the URL
response = requests.get(url)
logging.info(f"Sent GET request to: {url}")

# Create a BeautifulSoup object to parse the HTML content
soup = BeautifulSoup(response.content, "html.parser")

# Find all the listing items (up to 20)
listings = soup.find_all("li", class_="cl-static-search-result")[:20]
logging.info(f"Found {len(listings)} listings")

# Create a directory to store the downloaded images
os.makedirs("craigslist_images", exist_ok=True)
logging.info("Created directory: craigslist_images")

# Iterate over each listing
for listing in listings:
    # Extract the URL of the listing
    listing_url = listing.find("a")["href"]
    logging.info(f"Processing listing: {listing_url}")

    # Send a GET request to the listing URL
    listing_response = requests.get(listing_url)
    logging.info(f"Sent GET request to: {listing_url}")
    listing_soup = BeautifulSoup(listing_response.content, "html.parser")

    # Find all the large image elements in the listing
    large_images = listing_soup.find_all("a", class_="thumb")
    logging.info(f"Found {len(large_images)} large images in the listing")

    # Iterate over each large image and download it
    for i, image in enumerate(large_images, start=1):
        image_url = image["href"]
        image_response = requests.get(image_url)
        logging.info(f"Sent GET request to: {image_url}")

        # Generate a unique filename for the image
        filename = f"{listing_url.split('/')[-1].split('.')[0]}_image_{i}.jpg"

        # Save the image to the directory
        with open(os.path.join("craigslist_images", filename), "wb") as file:
            file.write(image_response.content)

        logging.info(f"Downloaded: {filename}")

        # Delay for a reasonable time between image downloads
        time.sleep(1)

    # Delay for a reasonable time between listing requests
    time.sleep(2)

logging.info("Image downloading completed.")
