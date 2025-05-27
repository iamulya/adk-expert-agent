# expert-agents/tools/marp_document_tools.py
import os
import asyncio
import logging
import subprocess
import tempfile
from typing import Optional

from google.adk.tools import ToolContext
from .marp_utils import (
    _check_marp_cli, _upload_to_gcs_and_get_signed_url,
    MARP_CLI_COMMAND, GENERATED_DOCS_SUBDIR, DOC_LINK_STATE_KEY
)

logger = logging.getLogger(__name__)

async def _generate_with_marp_and_upload(
    markdown_content: str,
    output_filename_base: str, # Base name without extension
    marp_format_flag: str, # --pdf, --html, --pptx
    gcs_content_type: str,
    file_extension: str, # .pdf, .html, .pptx
    tool_context: Optional[ToolContext] = None
) -> str:
    if not _check_marp_cli():
        return "Error: marp-cli is not installed or accessible. Cannot generate document."

    local_marp_output_dir = os.path.join(tempfile.gettempdir(), GENERATED_DOCS_SUBDIR)
    os.makedirs(local_marp_output_dir, exist_ok=True)
    
    safe_local_filename = "".join(c if c.isalnum() or c in ('.', '_', '-') else '_' for c in output_filename_base)
    local_output_path = os.path.join(local_marp_output_dir, safe_local_filename + file_extension)
    
    tmp_md_file_path = None
    try:
        with tempfile.NamedTemporaryFile(mode="w+", delete=False, suffix=".md", encoding='utf-8') as tmp_md_file:
            tmp_md_file.write(markdown_content)
            tmp_md_file_path = tmp_md_file.name
        
        cmd = [MARP_CLI_COMMAND, tmp_md_file_path, marp_format_flag, "-o", local_output_path, "--allow-local-files"]
        logger.info(f"Executing Marp command: {' '.join(cmd)}")
        result = subprocess.run(cmd, check=True, capture_output=True, text=True, encoding='utf-8')
        
        if os.path.exists(local_output_path):
            logger.info(f"Marp generated file locally: {local_output_path}. Marp stdout: {result.stdout}")
            return await _upload_to_gcs_and_get_signed_url(
                local_file_path=local_output_path,
                gcs_object_name_base=output_filename_base,
                content_type=gcs_content_type,
                file_extension=file_extension,
                tool_context=tool_context,
                markdown_content_for_fallback=markdown_content
            )
        else:
            logger.error(f"Marp file generation commanded, but output file '{local_output_path}' not found. Marp stdout: {result.stdout}. Stderr: {result.stderr}")
            return f"Marp file generation commanded, but output file not found. Marp output: {result.stdout}. Error: {result.stderr}"

    except subprocess.CalledProcessError as e:
        logger.error(f"Error running marp-cli for {marp_format_flag}: {e.stderr}", exc_info=True)
        return f"Error generating document with marp-cli: {e.stderr}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during {marp_format_flag} generation: {str(e)}", exc_info=True)
        return f"An unexpected error occurred during document generation: {str(e)}"
    finally:
        if tmp_md_file_path and os.path.exists(tmp_md_file_path):
            os.remove(tmp_md_file_path)
        if 'local_output_path' in locals() and os.path.exists(local_output_path):
             try:
                os.remove(local_output_path)
             except Exception as e_rem:
                 logger.warning(f"Could not remove temporary local Marp file {local_output_path}: {e_rem}")

async def generate_pdf_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pdf", "application/pdf", ".pdf", tool_context
    )

async def generate_html_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--html", "text/html", ".html", tool_context
    )

async def generate_pptx_slides_from_markdown_with_gcs(markdown_content: str, output_filename: str, tool_context: ToolContext) -> str:
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--pptx", "application/vnd.openxmlformats-officedocument.presentationml.presentation", ".pptx", tool_context
    )
