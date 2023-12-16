#!/usr/local/bin/python3.9

URL = "https://travel.state.gov/content/travel/en/legal/visa-law0/visa-bulletin/2023/visa-bulletin-for-july-2023.html"

import logging
import re
import sys
import requests
from bs4 import BeautifulSoup
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Bulletin Scrapper")

# Send a GET request
response = requests.get(URL)

# If the GET request is successful, the status code will be 200
if response.status_code == 200:
    logger.info("Request Successful")

    # Get the content of the response
    page_content = response.content

    # Create a BeautifulSoup object and specify the parser
    soup = BeautifulSoup(page_content, "html.parser")
    tables = soup.find_all("table")

    dates = []
    for table in tables:
        row = table.find("tr").find("td")
        if not row:
            continue
        cell = re.sub("[^a-zA-Z]", "", str(row.text)).lower()
        if "employmentbased" in cell:
            country = table.find_all("tr")[0].find_all("td")[3].text
            if re.sub("[^a-zA-Z]", "", str(country)).lower() != "india":
                logger.error("Country found: {}".format(str(country)))
                raise ValueError("Invalid Country")
            dates.append(table.find_all("tr")[1].find_all("td")[3].text)

    # Create the body of your email
    email_body = " ".join(
        "Final action date: {0}. Date for filing application: {1}".format(
            dates[0], dates[1]
        )
    )

    # Set up your email
    msg = MIMEMultipart()
    msg["From"] = "rmalhan0112@gmail.com"
    msg["To"] = "rmalhan0112@gmail.com"
    msg["Subject"] = "Automated Visa Bulletin Update"
    msg.attach(MIMEText(email_body, "plain"))

    # Set up your server
    server = smtplib.SMTP("smtp.gmail.com", 587)
    server.starttls()

    # Your email account login credentials
    server.login("rmalhan0112@gmail.com", "lfxsfwikrqeepcjn")

    # Send the email
    server.send_message(msg)
    server.quit()
