"""
Demo: Using Callbacks with UniParserClient

This demo shows how to submit an asynchronous parsing task with a callback URL.
When the parsing is complete, the UniParser service will send a POST request
to the specified callback_url with the results.
"""

import os

from uniparser_tools.api.clients import UniParserClient
from uniparser_tools.common.constant import ParseMode, ParseModeTextual


def main():
    # 1. Initialize the client
    # Make sure to set your API key in the environment variables
    api_key = os.getenv("UNIPARSER_API_KEY")
    if not api_key:
        print("Please set the UNIPARSER_API_KEY environment variable.")
        return

    host = "https://uniparser.dp.tech/"  # Replace with your actual host if different
    parser = UniParserClient(host=host, api_key=api_key)

    # 2. Define callback parameters
    # In a real scenario, this would be a URL to your server that can receive POST requests
    callback_url = "http://your-callback-server.com/api/v1/callback"
    callback_secret = "your-shared-secret-for-verification"

    # 3. Path to the PDF file to be parsed
    pdf_path = "path/to/your/document.pdf"
    if not os.path.exists(pdf_path):
        print(f"File not found: {pdf_path}. Please provide a valid PDF path.")
        # For demonstration purposes, we'll continue, but the actual request would fail.
        # return

    # 4. Trigger asynchronous parsing with callback
    print(f"Triggering asynchronous parsing for: {pdf_path}")
    print(f"Callback URL: {callback_url}")

    result = parser.trigger_file(
        file_path=pdf_path,
        sync=False,  # Must be False for async callbacks to be meaningful
        textual=ParseModeTextual.DigitalExported,
        table=ParseMode.OCRFast,
        callback_url=callback_url,
        callback_secret=callback_secret,
    )

    if result["status"] == "success":
        token = result["token"]
        print("Task submitted successfully!")
        print(f"Token: {token}")
        print("\nUniParser will now process the file in the background.")
        print("Once finished, it will POST the result to your callback URL.")
        print("Verify X-UniParser-Signature over the exact raw callback body using your callback_secret.")
    else:
        print(f"Failed to submit task: {result.get('message')}")
        if "description" in result:
            print(f"Details: {result['description']}")


if __name__ == "__main__":
    main()
