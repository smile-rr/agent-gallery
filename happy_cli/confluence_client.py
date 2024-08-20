import logging
import os
from bs4 import BeautifulSoup
from dotenv import load_dotenv
import requests
import base64
from typing import List, Dict
from pydantic import BaseModel, Field

class ConfluenceSearchInput(BaseModel):
    keywords: List[str] = Field(..., description="List of keywords to search for in Confluence.")


class ConfluenceSearchResult(BaseModel):
    url: str = Field(..., description="URL of the Confluence page.")
    title: str = Field(..., description="Title of the Confluence page.")
    content: str = Field(..., description="Content of the Confluence page.")

class ConfluenceSearchOutput(BaseModel):
    results: List[ConfluenceSearchResult] = Field(..., description="List of search results from Confluence.")
FUNCTION_DESCRIPTIONS = {
    "confluence_search": {
        "name": "confluence_search",
        "description": "Searches Confluence for pages containing the specified keywords.",
        "parameters": ConfluenceSearchInput.schema(),
        "returns": ConfluenceSearchOutput.schema(),
    }
}

class ConfluenceClient:
    def __init__(self):
        # Fix the base_url definition by removing the comma
        self.base_url = os.getenv("CONFLUENCE_URL")
        self.username = os.getenv("CONFLUENCE_USERNAME")
        self.api_token = os.getenv("CONFLUENCE_TOKEN")
        self.headers = {"Authorization": f"Basic {self._get_basic_auth_token()}",
                        "Content-Type": "application/json"}

    def _get_basic_auth_token(self):
        credentials = f"{self.username}:{self.api_token}".encode("utf-8")
        return base64.b64encode(credentials).decode("utf-8")
    def search(self, keywords: List[str]) -> List[ConfluenceSearchResult]:
        # Search Confluence using the provided keywords
        # This is a simplified example
        results = self.client.search_by_title_and_content(keywords)
        search_results = []
        for result in results:
            search_result = ConfluenceSearchResult(
                url=result['url'],
                title=result['title'],
                content=result['content']
            )
            search_results.append(search_result)
        return search_results
    def global_search(self, keys, start=0, limit=50):
        # Define the API URL and parameters
        url = f"{self.base_url}/rest/api/search"
        
        # Properly format the CQL query
        cql_title_body = " OR ".join([f"title ~ '{key}' OR text ~ '{key}'" for key in keys])
        params = {
            "cql": f"{cql_title_body}",
            "start": start,
            "limit": limit
        }

        # URL-encode the CQL query
        # params["cql"] = urllib.parse.quote(params["cql"])

        # Make the request
        response = requests.get(url, headers=self.headers, params=params)
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()

    def search_by_title_and_content(self, query, start=0, limit=50):
        """
        Searches for documents by title and content, fetching each page and constructing the response with the URL, title, and content text.

        :param query: The search query.
        :param start: The starting index of the results to return.
        :param limit: The maximum number of results to return.
        :return: A list of dictionaries with 'url', 'title', and 'content_text' keys or None if an error occurs.
        """
        results = self.global_search(query, start, limit)

        if results is None:
            return None

        extracted_results = []

        for result in results.get("results", [])[:1]:
            page_id = result["content"]["id"]

            # Fetch the page data
            page_info = self.get_page(page_id)
            print(page_info)
            if page_info:
                # Construct the URL
                url = f"{self.base_url}/pages/viewpage.action?pageId={page_id}"
                title = page_info.get("title")
                content = page_info['body']['storage']['value']
                soup = BeautifulSoup(content, 'html.parser')
                content_text = soup.get_text()
                extracted_results.append({
                    "url": url,
                    "title": title,
                    "content": content_text
                })

        return extracted_results
    def get_page(self, page_id):
        """
        Retrieves a Confluence page by its ID and constructs the URL for that page.

        :param page_id: The ID of the Confluence page.
        :return: A dictionary containing the page data and the constructed URL.
        """
        # Construct the request URL
        url = f"{self.base_url}/rest/api/content/{page_id}?expand=body.storage"
        response = requests.get(url, headers=self.headers)
        response.raise_for_status()
        return response.json()        

# Example usage
if __name__ == "__main__":
    load_dotenv(".env.local")
    client = ConfluenceClient()
    # Assuming `client` is an instance of your Confluence client class
    query = ["google"]

    # Search by title and content
    search_results = client.search_by_title_and_content(query)

    if search_results:
        for result in search_results:
            print("URL:", result["url"])
            print("Title:", result["title"])
            print("Content Text:", result["content_text"])
            print()
    else:
        print("No results found.")