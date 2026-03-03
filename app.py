"""
PDF to JSON Extraction API
A FastAPI-based service for extracting structured content from PDFs including text, tables, and links.

Dependencies:
pip install fastapi uvicorn pdfplumber python-multipart pymupdf openai
"""

# import _typeshed.importlib
from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pdfplumber
import fitz  # PyMuPDF for links and metadata
import io
import re
import base64
import os
import json
import tempfile
import sys
from typing import List, Dict, Any, Optional
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Ensure user site-packages are on the path (needed when uvicorn --reload
# spawns a fresh subprocess that may not inherit the full PYTHONPATH).
import site
for _sp in site.getusersitepackages() if isinstance(site.getusersitepackages(), list) else [site.getusersitepackages()]:
    if _sp not in sys.path:
        sys.path.insert(0, _sp)

import httpx
import openpyxl
from google import genai
import dotenv
app = FastAPI(
    title="PDF to JSON Extraction API",
    description="Extract complete PDF content including text, tables, and links",
    version="1.0.0"
)
dotenv.load_dotenv()
# --- Configuration ---
# Gemini API Key configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    print("Warning: GEMINI_API_KEY not set in environment.")


class PDFExtractor:
    """Efficient PDF content extractor with support for text, tables, and links."""

    def __init__(self):
        self.executor = ThreadPoolExecutor(max_workers=4)

    def _clean_text(self, text: str) -> str:
        """Remove Hindi characters, (cid:...) artifacts, and extra whitespace."""
        if not text:
            return ""
        # Remove (cid:...) artifacts
        text = re.sub(r'\(cid:\d+\)', '', text)
        # Remove Hindi characters
        text = re.sub(r'[\u0900-\u097F]+', '', text)
        # Remove empty parentheses artifacts like () or ( )
        text = re.sub(r'\(\s*\)', '', text)
        # Remove potential nested empty parentheses like ( ( ) )
        text = re.sub(r'\(\s*\(\s*\)\s*\)', '', text)
        # Remove leading slashes if they precede alphanumeric chars (common artifact)
        text = re.sub(r'(^|\s)/+(?=[a-zA-Z0-9])', r'\1', text)
        # Remove multiple spaces/newlines
        text = re.sub(r'\s+', ' ', text).strip()
        # Final cleanup for empty parens at ends
        text = re.sub(r'^\(\s*\)$', '', text)
        return text

    def extract_text_blocks(self, page) -> List[Dict[str, Any]]:
        """Extract text with positioning information, filtering out Hindi."""
        text_blocks = []
        words = page.extract_words(
            x_tolerance=3,
            y_tolerance=3,
            keep_blank_chars=False,
            use_text_flow=True
        )

        # Group words into lines
        lines = {}
        for word in words:
            # Clean Hindi characters from the word text instead of skipping the whole word
            original_text = word['text']
            cleaned_word_text = self._clean_text(original_text)
            
            # If nothing remains (was pure Hindi or artifact), skip it
            if not cleaned_word_text:
                continue

            # Update the word text with cleaned version
            word['text'] = cleaned_word_text
            
            y_pos = round(word['top'], 1)
            if y_pos not in lines:
                lines[y_pos] = []
            lines[y_pos].append(word)

        # Create text blocks from lines
        for y_pos in sorted(lines.keys()):
            line_words = sorted(lines[y_pos], key=lambda w: w['x0'])
            text = ' '.join(w['text'] for w in line_words)
            # Re-clean the joined line to handle any artifacts formed by joining
            cleaned_text = self._clean_text(text)
            
            if cleaned_text:
                text_blocks.append({
                    'text': cleaned_text,
                    'position': {
                        'x': float(line_words[0]['x0']),
                        'y': float(y_pos),
                        'width': float(line_words[-1]['x1'] - line_words[0]['x0']),
                        'height': float(line_words[0]['bottom'] - line_words[0]['top'])
                    }
                })

        return text_blocks

    def extract_tables(self, page) -> List[Dict[str, Any]]:
        """Extract tables with cell data."""
        tables = []
        table_settings = {
            "vertical_strategy": "lines",
            "horizontal_strategy": "lines",
            "intersection_tolerance": 3,
        }

        extracted_tables = page.find_tables(table_settings=table_settings)

        for idx, table in enumerate(extracted_tables):
            table_data = table.extract()
            if table_data:
                # Clean and structure table data
                # Clean headers
                headers = [self._clean_text(h) if h else "" for h in (table_data[0] if table_data else [])]
                rows = table_data[1:] if len(table_data) > 1 else []

                structured_rows = []
                for row in rows:
                    if any(cell for cell in row if cell):  # Skip empty rows
                        row_dict = {}
                        for col_idx, cell in enumerate(row):
                            header = headers[col_idx] if col_idx < len(headers) else f"Column_{col_idx}"
                            # Clean cell content
                            cell_content = self._clean_text(cell) if cell else ""
                            key = header if header else f"Column_{col_idx}"
                            row_dict[key] = cell_content
                        structured_rows.append(row_dict)

                # Convert bbox tuple to serializable dict
                bbox = table.bbox if table.bbox else (0, 0, 0, 0)
                tables.append({
                    'table_index': idx,
                    'headers': headers,
                    'rows': structured_rows,
                    'bbox': {
                        'x0': float(bbox[0]),
                        'y0': float(bbox[1]),
                        'x1': float(bbox[2]),
                        'y1': float(bbox[3])
                    }
                })

        return tables

    def extract_links(self, pdf_bytes: bytes, page_num: int) -> List[Dict[str, Any]]:
        """Extract hyperlinks using PyMuPDF, including anchor text."""
        links = []
        try:
            # We open a new document instance for thread safety when accessing links
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            page = doc[page_num]

            for link in page.get_links():
                # Convert bbox to serializable dict
                bbox = link.get('from', (0, 0, 0, 0))

                # Extract text covered by the link (anchor text)
                anchor_text = page.get_textbox(bbox).strip()
                # Clean anchor text
                anchor_text = self._clean_text(anchor_text)

                link_data = {
                    'type': 'unknown',
                    'anchor_text': anchor_text,
                    'bbox': {
                        'x0': float(bbox[0]) if isinstance(bbox, (tuple, list)) and len(bbox) > 0 else 0,
                        'y0': float(bbox[1]) if isinstance(bbox, (tuple, list)) and len(bbox) > 1 else 0,
                        'x1': float(bbox[2]) if isinstance(bbox, (tuple, list)) and len(bbox) > 2 else 0,
                        'y1': float(bbox[3]) if isinstance(bbox, (tuple, list)) and len(bbox) > 3 else 0
                    }
                }

                # Handle different link kinds
                if link['kind'] == fitz.LINK_URI:
                    link_data['type'] = 'external'
                    link_data['url'] = link['uri']
                elif link['kind'] == fitz.LINK_GOTO:
                    link_data['type'] = 'internal'
                    if 'page' in link and link['page'] >= 0:
                        link_data['page'] = link['page']
                        link_data['target_page_label'] = link['page'] + 1
                elif link['kind'] == fitz.LINK_NAMED:
                    link_data['type'] = 'internal'
                    # Try to resolve named destination
                    try:
                        # Some versions of PyMuPDF resolve this automatically in 'page' key
                        # If not, we might need doc.resolve_link(link) but get_links usually handles it
                        if 'page' in link and link['page'] >= 0:
                            link_data['page'] = link['page']
                            link_data['target_page_label'] = link['page'] + 1
                        elif 'name' in link:
                            link_data['name'] = link['name']
                    except Exception:
                        pass
                elif link['kind'] == fitz.LINK_LAUNCH:
                    link_data['type'] = 'launch'
                    if 'file' in link:
                        link_data['url'] = link['file']  # Use url field for file path too
                elif link['kind'] == fitz.LINK_GOTOR:
                    link_data['type'] = 'remote_goto'
                    if 'file' in link:
                        link_data['url'] = link['file']
                    if 'page' in link and link['page'] >= 0:
                        link_data['page'] = link['page']
                        link_data['target_page_label'] = link['page'] + 1

                # Always add the link, even if target is not fully resolved, 
                # effectively reporting "broken" or "scroll-to" links as well
                links.append(link_data)

            doc.close()
        except Exception as e:
            # Silently handle if PyMuPDF fails
            pass

        return links

    def detect_structure(self, text_blocks: List[Dict]) -> Dict[str, Any]:
        """Detect document structure like headers, lists, etc."""
        structure = {
            'headers': [],
            'lists': [],
            'key_value_pairs': []
        }

        for block in text_blocks:
            text = block['text']

            # Detect headers (all caps, short lines, or ending with colon)
            if (text.isupper() and len(text.split()) <= 10) or \
               (text.endswith(':') and len(text) < 100):
                structure['headers'].append(block)

            # Detect list items
            if re.match(r'^\d+\.|\-|\*|\•', text.strip()):
                structure['lists'].append(block)

            # Detect key-value pairs with better logic
            colon_match = re.match(r'^([^:]{1,80}):\s*(.+)', text)
            slash_match = re.match(r'^([^/]{1,80})/\s*(.+)', text)

            match = colon_match or slash_match
            if match and len(match.group(2).strip()) > 1:
                key = match.group(1).strip()
                value = match.group(2).strip()

                # Additional validation: key shouldn't be too long or contain multiple sentences
                if len(key) < 100 and key.count('.') < 2:
                    structure['key_value_pairs'].append({
                        'key': key,
                        'value': value,
                        'position': {
                            'x': float(block['position']['x']),
                            'y': float(block['position']['y']),
                            'width': float(block['position']['width']),
                            'height': float(block['position']['height'])
                        }
                    })

        return structure

    def extract_embedded_files(self, pdf_bytes: bytes) -> List[Dict[str, Any]]:
        """Extract embedded files/attachments from PDF."""
        embedded_files = []
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")

            # Check for embedded files
            if hasattr(doc, 'embfile_count') and doc.embfile_count() > 0:
                for i in range(doc.embfile_count()):
                    try:
                        file_info = doc.embfile_info(i)
                        file_data = {
                            'index': i,
                            'name': file_info.get('name', f'attachment_{i}'),
                            'filename': file_info.get('filename', ''),
                            'description': file_info.get('desc', ''),
                            'size': file_info.get('size', 0),
                            'creation_date': file_info.get('creationDate', ''),
                            'modification_date': file_info.get('modDate', '')
                        }

                        # Optionally extract file content (for small files)
                        # We increased limit to 5MB and enable base64 extraction
                        if file_info.get('size', 0) < 5242880:  # Less than 5MB
                             content = doc.embfile_get(i)
                             file_data['content_base64'] = base64.b64encode(content).decode('utf-8')

                        embedded_files.append(file_data)
                    except Exception as e:
                        continue

            doc.close()
        except Exception as e:
            pass

        return embedded_files

    def extract_metadata(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract PDF metadata."""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            metadata = {
                'title': doc.metadata.get('title', ''),
                'author': doc.metadata.get('author', ''),
                'subject': doc.metadata.get('subject', ''),
                'creator': doc.metadata.get('creator', ''),
                'producer': doc.metadata.get('producer', ''),
                'creation_date': doc.metadata.get('creationDate', ''),
                'modification_date': doc.metadata.get('modDate', ''),
                'page_count': doc.page_count
            }
            doc.close()
            return metadata
        except:
            return {}

    def process_page(self, page_data: tuple) -> Dict[str, Any]:
        """Process a single page."""
        page, page_num, pdf_bytes = page_data

        # Extract different components
        text_blocks = self.extract_text_blocks(page)
        tables = self.extract_tables(page)
        links = self.extract_links(pdf_bytes, page_num)
        structure = self.detect_structure(text_blocks)

        # Get raw text and clean it
        raw_text = page.extract_text() or ""
        raw_text = self._clean_text(raw_text)

        return {
            'page_number': page_num + 1,
            'raw_text': raw_text,
            'text_blocks': text_blocks,
            'tables': tables,
            'links': links,
            'structure': structure,
            'dimensions': {
                'width': float(page.width),
                'height': float(page.height)
            }
        }

    async def extract_from_pdf(self, pdf_bytes: bytes) -> Dict[str, Any]:
        """Extract all content from PDF."""
        result = {
            'status': 'success',
            'metadata': {},
            'embedded_files': [],
            'pages': [],
            'total_pages': 0,
            'summary': {
                'total_text_blocks': 0,
                'total_tables': 0,
                'total_links': 0,
                'total_embedded_files': 0
            }
        }

        try:
            # Extract metadata
            result['metadata'] = self.extract_metadata(pdf_bytes)

            # Extract embedded files
            embedded_files = self.extract_embedded_files(pdf_bytes)
            result['embedded_files'] = embedded_files
            result['summary']['total_embedded_files'] = len(embedded_files)

            # Process pages
            with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
                result['total_pages'] = len(pdf.pages)

                # Prepare page data for parallel processing
                page_data_list = [
                    (page, idx, pdf_bytes)
                    for idx, page in enumerate(pdf.pages)
                ]

                # Process pages in parallel
                loop = asyncio.get_event_loop()
                with ThreadPoolExecutor(max_workers=4) as executor:
                    futures = [
                        loop.run_in_executor(executor, self.process_page, page_data)
                        for page_data in page_data_list
                    ]
                    pages = await asyncio.gather(*futures)

                result['pages'] = pages

                # Calculate summary
                for page in pages:
                    result['summary']['total_text_blocks'] += len(page['text_blocks'])
                    result['summary']['total_tables'] += len(page['tables'])
                    result['summary']['total_links'] += len(page['links'])

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

        return result


# Initialize extractor
extractor = PDFExtractor()


# ---------------------------------------------------------------------------
# External file download + summarization helper
# ---------------------------------------------------------------------------

def _extract_excel_text(file_bytes: bytes) -> str:
    """Extract readable text from an Excel (.xlsx) file."""
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), data_only=True)
        lines = []
        for sheet in wb.worksheets:
            lines.append(f"[Sheet: {sheet.title}]")
            for row in sheet.iter_rows(values_only=True):
                non_empty = [str(c) for c in row if c is not None and str(c).strip()]
                if non_empty:
                    lines.append(" | ".join(non_empty))
        return "\n".join(lines)
    except Exception as e:
        return f"[Could not parse Excel: {e}]"


def _extract_text_from_bytes(file_bytes: bytes, content_type: str, url: str) -> str:
    """Detect file type and return extracted plain text."""
    lower_url = url.lower()
    is_excel = (
        "spreadsheet" in content_type
        or lower_url.endswith(".xlsx")
        or lower_url.endswith(".xls")
    )
    is_pdf = "pdf" in content_type or lower_url.endswith(".pdf")

    if is_excel:
        return _extract_excel_text(file_bytes)

    if is_pdf:
        # Run sync extractor in event loop — already done inside async context
        import asyncio
        loop = asyncio.get_event_loop()
        # extract_from_pdf is async; call it synchronously via a new task
        # We'll handle this in the async caller instead
        return "__PDF__"  # sentinel handled in async caller

    # Fallback: try to decode as UTF-8 text
    try:
        return file_bytes.decode("utf-8", errors="replace")
    except Exception:
        return "[Binary file — content not readable as text]"


async def _get_gemini_summary(client, text: str, filename_hint: str) -> str:
    """Ask Gemini to summarize file content in 3-5 sentences."""
    prompt = (
        f"You are summarizing a document referenced in a government bid/tender.\n"
        f"File hint: {filename_hint}\n\n"
        f"Summarize the following content in 3-5 concise sentences. "
        f"Focus on: purpose of the document, key items/quantities/prices (if any), "
        f"important clauses or conditions. Do NOT output JSON, just plain text.\n\n"
        f"--- FILE CONTENT (first 8000 chars) ---\n{text[:8000]}"
    )
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text.strip()
    except Exception as e:
        return f"[Gemini summarization failed: {e}]"


async def summarize_external_files(
    file_refs: Dict[str, Any],
    gemini_client,
    cookies: Optional[Dict[str, str]] = None,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """
    For each entry in external_file_references:
      - If the value is a URL  → download (with optional auth cookies), extract text, summarize with Gemini
      - Otherwise             → return {url: null, summary: "No downloadable link found."}
    Returns a dict with the same keys but values replaced by {url, summary} objects.
    """
    enriched = {}

    default_headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "application/pdf,application/octet-stream,*/*",
        "Referer": "https://bidplus.gem.gov.in/",
    }
    if extra_headers:
        default_headers.update(extra_headers)

    cookie_jar = httpx.Cookies(cookies or {})

    async with httpx.AsyncClient(
        timeout=30.0,
        follow_redirects=True,
        headers=default_headers,
        cookies=cookie_jar,
    ) as http:
        for key, value in file_refs.items():
            if not value or value in ("Not Found", "null", "N/A"):
                enriched[key] = {
                    "url": None,
                    "summary": "No downloadable link found."
                }
                continue

            # Check if value looks like a URL
            if not (isinstance(value, str) and value.startswith("http")):
                enriched[key] = {
                    "url": None,
                    "summary": f"Reference noted (not a direct URL): {value}"
                }
                continue

            url = value
            try:
                resp = await http.get(url)
                resp.raise_for_status()
                file_bytes = resp.content
                content_type = resp.headers.get("content-type", "").lower()

                # Check for PDF sentinel
                lower_url = url.lower()
                is_pdf = "pdf" in content_type or lower_url.endswith(".pdf")

                if is_pdf:
                    pdf_result = await extractor.extract_from_pdf(file_bytes)
                    pages_text = "\n".join(
                        p.get("raw_text", "") for p in pdf_result.get("pages", [])
                    )
                    text_content = pages_text
                else:
                    text_content = _extract_text_from_bytes(file_bytes, content_type, url)

                if not text_content.strip():
                    text_content = "[File downloaded but appears to be empty or unreadable.]"

                summary = await _get_gemini_summary(gemini_client, text_content, key)
                enriched[key] = {"url": url, "summary": summary}

            except httpx.HTTPStatusError as e:
                enriched[key] = {
                    "url": url,
                    "summary": f"Failed to download (HTTP {e.response.status_code})."
                }
            except Exception as e:
                enriched[key] = {
                    "url": url,
                    "summary": f"Error processing file: {str(e)}"
                }

    return enriched


@app.post("/extract-pdf", response_class=JSONResponse)
async def extract_pdf(file: UploadFile = File(...)):
    """
    Extract complete content from a PDF file.
    
    Returns:
    - status: 'success'
    - metadata: PDF metadata (title, author, etc.)
    - pages: Array of page objects with text, tables, links
    - embedded_files: Array of embedded files
    - summary: Statistics about extracted content
    """

    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file
    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # Extract content
    result = await extractor.extract_from_pdf(pdf_bytes)

    return JSONResponse(content=result)


@app.post("/extract-pdfs-batch", response_class=JSONResponse)
async def extract_pdfs_batch(files: List[UploadFile] = File(...)):
    """
    Extract content from multiple PDF files in batch.
    
    Returns:
    - Array of extraction results for each file
    """

    if len(files) > 50:
        raise HTTPException(
            status_code=400,
            detail="Maximum 50 files allowed per batch"
        )

    results = []

    for file in files:
        try:
            if not file.filename.lower().endswith('.pdf'):
                results.append({
                    'filename': file.filename,
                    'status': 'error',
                    'error': 'Not a PDF file'
                })
                continue

            pdf_bytes = await file.read()
            extraction_result = await extractor.extract_from_pdf(pdf_bytes)

            results.append({
                'filename': file.filename,
                'status': 'success',
                'data': extraction_result
            })

        except Exception as e:
            results.append({
                'filename': file.filename,
                'status': 'error',
                'error': str(e)
            })

    return JSONResponse(content={'results': results})


@app.post("/extract-bid-gemini", response_class=JSONResponse)
async def extract_bid_gemini(
    file: UploadFile = File(...),
    cookie_string: Optional[str] = Form(
        default=None,
        description=(
            "Optional: raw browser Cookie header string from an authenticated GeM session. "
            "Paste the value of the 'Cookie' header from your browser DevTools while logged in to GeM. "
            "Example: 'PHPSESSID=abc123; other_cookie=xyz'"
        )
    ),
):
    """
    Extract structured bid information using Google Gemini 2.5 Flash API.

    Optionally pass `cookie_string` (multipart form field) with your GeM browser
    session cookies to allow downloading of restricted file attachments.
    Requires GEMINI_API_KEY environment variable.
    """
    
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(status_code=400, detail="Only PDF files are supported")

    # Read file
    pdf_bytes = await file.read()

    if len(pdf_bytes) == 0:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    # 1. Extract raw text and links using existing extractor
    extraction_result = await extractor.extract_from_pdf(pdf_bytes)
    
    # 2. Prepare context for Gemini
    full_text = ""
    all_links = []
    
    for page in extraction_result.get('pages', []):
        full_text += f"--- Page {page['page_number']} ---\n"
        full_text += page.get('raw_text', '') + "\n\n"
        
        for link in page.get('links', []):
            link_info = f"Page {page['page_number']} Link: '{link.get('anchor_text', '')}' -> {link.get('url', 'Internal/Other')}"
            all_links.append(link_info)
            
    embedded_files_info = []
    for ef in extraction_result.get('embedded_files', []):
        embedded_files_info.append(f"Embedded File: {ef.get('name')} ({ef.get('filename')})")

    # 3. Construct Prompt
    system_prompt = """You are an expert data extraction assistant. Your task is to extract specific bid information from the provided PDF text and links.
    You must output strictly valid JSON format without any markdown formatting or explanations.
    
    Extract the data into the following JSON structure:

    {
      "bid_metadata": {
        "bid_number": "Extract Bid Number",
        "dated": "Extract Date",
        "bid_status": "Extract Status (e.g. Active)",
        "bid_end_date_time": "Extract End Date/Time",
        "bid_opening_date_time": "Extract Opening Date/Time",
        "bid_offer_validity": "Extract Offer Validity"
      },
      "authority_details": {
        "ministry": "Extract Ministry",
        "department": "Extract Department",
        "organisation": "Extract Organisation",
        "beneficiary_name": "Extract Beneficiary Name",
        "beneficiary_full_address": "Extract Beneficiary Full Address"
      },
      "item_details": {
        "category": "Extract Category",
        "item_description": "Extract Item Description",
        "contract_period": "Extract Contract Period",
        "project_based_quantity": "Extract Project Based Quantity (number)"
      },
      "eligibility_criteria": {
        "min_average_annual_turnover_3_years": "Extract Turnover",
        "past_experience_required": "Extract Experience",
        "mse_relaxation": "Extract MSE Relaxation (Yes/No)",
        "startup_relaxation": "Extract Startup Relaxation (Yes/No)",
        "documents_required_from_seller": [
          "List of required documents"
        ]
      },
      "financial_terms": {
        "evaluation_method": "Extract Evaluation Method",
        "emd_detail": {
          "advisory_bank": "Extract Advisory Bank",
          "amount": "Extract Amount (number)",
          "fdr_pledged_to": "Extract FDR Pledged To"
        },
        "epbg_detail": {
          "advisory_bank": "Extract Advisory Bank",
          "percentage": "Extract Percentage (number)",
          "duration_months": "Extract Duration (number)"
        },
        "payment_terms": "Extract Payment Terms"
      },
      "descriptive_paragraphs": {
        "turnover_and_experience_clause": "Extract full clause text",
        "mse_purchase_preference_clause": "Extract full clause text",
        "labour_code_compliance": "Extract full clause text",
        "land_border_restriction": "Extract full clause text"
      },
      "buyer_added_atc_clauses": [
        "List of buyer added ATC clauses"
      ],
      "external_file_references": {
        "excel_price_breakup": "Extract filename/link",
        "scope_of_work_pdf": "Extract filename/link",
        "terms_and_conditions_pdf": "Extract filename/link",
        "additional_repair_details_pdf": "Extract filename/link",
        "buyer_uploaded_atc_document": "Extract filename/link"
      }
    }
       
    Refine the data to be clean and precise. If a field is not found, use null or string "Not Found".
    """

    user_message = f"""Here is the content extracted from the PDF Bid Document:

    [LINKS AND EMBEDDED FILES]
    {chr(10).join(all_links)}
    {chr(10).join(embedded_files_info)}
    
    [DOCUMENT TEXT]
    {full_text}
    
    Please parse this and return the JSON object with the 8 sections requested."""

    try:
        # Initialize Gemini client
        client = genai.Client(api_key=GEMINI_API_KEY)
        
        # Combine system prompt and user message for context
        # Gemini 2.5 Flash works well with direct prompts or messages structure
        
        response = client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=f"{system_prompt}\n\n{user_message}",
            config={
                'response_mime_type': 'application/json'
            }
        )
        
        # Parse JSON response
        parsed_json = json.loads(response.text)

        # Parse cookie_string into a dict for authenticated downloads
        cookies_dict: Optional[Dict[str, str]] = None
        if cookie_string and cookie_string.strip():
            cookies_dict = {}
            for part in cookie_string.split(";"):
                part = part.strip()
                if "=" in part:
                    k, _, v = part.partition("=")
                    cookies_dict[k.strip()] = v.strip()

        # Enrich external_file_references with download + Gemini summaries
        if "external_file_references" in parsed_json and isinstance(
            parsed_json["external_file_references"], dict
        ):
            parsed_json["external_file_references"] = await summarize_external_files(
                parsed_json["external_file_references"], client, cookies=cookies_dict
            )

        return JSONResponse(content=parsed_json)

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing with Gemini: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)