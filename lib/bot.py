from langchain_nvidia_ai_endpoints import ChatNVIDIA
from dotenv import load_dotenv, find_dotenv
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate, ChatPromptTemplate


def load_worker_chain(modelName='meta/llama-3.1-405b-instruct'):
    
    """
        this function initializes a worker chain which is catered to extract the HTML page into an structured JSON form.
        Make sure that you have the API stored in an .env file under the name:

            NVIDIA_API_KEY = 'YOUR_API'

        params : modelName, should be available in the ChatNVIDIA

        returns : chain --> langchain chain, that could be use as 

            ```
            worker = load_worker_chain()
            response = worker.invoke({
                'html_content' : htmlpage
            })
            ```
    """



    llm = ChatNVIDIA(
        model_name=modelName,
        max_tokens=None,
        max_retries=3,
        temperature=0.1,
        timeout=None
    )

    system_template = """
    # HTML Content Scraper Bot

    You are an AI assistant designed to extract and structure content from HTML. Your task is to analyze the provided HTML body and return clean, highly structured data in a single, comprehensive JSON format. Follow these guidelines:

    1. Input: You will receive the HTML body of a web page.

    2. Output: Provide a single JSON object containing all structured data extracted from the HTML.

    3. Extraction Rules:
    - Identify and extract the main content, ignoring navigation menus, footers, and sidebars.
    - Extract the page title and any global metadata.
    - Identify and extract all products or main sections, including their titles, descriptions, features, associated images, and any hyperlinks or attachment links.
    - Preserve the hierarchical structure of the content.
    - Extract all relevant hyperlinks, including their text and URLs.
    - Identify and extract any attachment links, such as PDFs or other downloadable files.

    4. Data Cleaning:
    - Remove any HTML tags from the extracted text, except for hyperlinks which should be preserved in a structured format.
    - Decode HTML entities (e.g., &amp; to &, &quot; to ").
    - Trim leading and trailing whitespace from all extracted text.
    - Normalize whitespace within text (replace multiple spaces with a single space).

    5. Output Structure:
    Provide the extracted data in the following JSON format:

    {{
        "title": "string",
        "metadata": {{
        "author": "string or null",
        "date": "string or null",
        "tags": ["string"]
        }},
        "products": [
        {{
            "title": "string",
            "description": "string",
            "features": [
            {{
                "type": "string (e.g., 'unordered')",
                "items": [
                {{
                    "text": "string",
                    "links": [
                    {{
                        "text": "string",
                        "url": "string"
                    }}
                    ]
                }}
                ]
            }}
            ],
            "images": [
            {{
                "url": "string",
                "alt_text": "string"
            }}
            ],
            "attachments": [
            {{
                "name": "string",
                "url": "string"
            }}
            ],
            "links": [
            {{
                "text": "string",
                "url": "string"
            }}
            ]
        }}
        ],
        "global_links": [
        {{
            "text": "string",
            "url": "string"
        }}
        ]
    }}

    6. Error Handling:
    - If you encounter any issues parsing the HTML or extracting content, include an "errors" field in the JSON output with relevant error messages.

    7. Additional Notes:
    - Ensure all relevant information, including hyperlinks and attachment links, is captured in a single JSON structure.
    - If certain fields are not present for some products, include them as null or empty arrays/objects as appropriate.
    - Preserve the original order of content elements as they appear in the HTML.

    Remember, your goal is to provide a single, comprehensive, and highly structured JSON that accurately represents all the main content of the web page, including all relevant links and attachments, making it easy for further processing or analysis.
    """

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)

    instruction_template = "HTML content to scrape:\n{html_content}\n\nPlease provide a single, comprehensive, and highly structured JSON output based on the above HTML content, following the structure and guidelines provided. Include all relevant hyperlinks and attachment links."
    instruction_message_prompt = HumanMessagePromptTemplate.from_template(instruction_template)

    prompt = ChatPromptTemplate.from_messages([system_message_prompt, instruction_message_prompt])

    chain = prompt | llm

    return chain
