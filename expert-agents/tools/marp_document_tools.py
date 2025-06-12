"""
This module contains the high-level functions for generating different types
of documents (PDF, HTML, PPTX) from Markdown content using Marp CLI.

Each function orchestrates the process by:
1.  Checking if the `marp-cli` is available.
2.  Writing the Markdown content to a temporary file.
3.  Executing the `marp-cli` as a subprocess with the correct flags.
4.  Handling the output and calling a utility function to upload the result to GCS.
5.  Managing errors and cleanup of temporary files.
"""

import os
import logging
import subprocess
import tempfile
from typing import Optional

from google.adk.tools import ToolContext
from .marp_utils import (
    _check_marp_cli,
    _upload_to_gcs_and_get_signed_url,
    MARP_CLI_COMMAND,
    GENERATED_DOCS_SUBDIR,
    DOC_LINK_STATE_KEY,
)

logger = logging.getLogger(__name__)

# A consistent prefix for user-facing error messages.
MARP_GENERATION_ERROR_MESSAGE_PREFIX = (
    "There was an error during the document generation process."
)


def _handle_marp_error_in_context(
    tool_context: Optional[ToolContext],
    error_message: str,
):
    """
    Helper to manage state when a Marp generation error occurs.

    It saves the error message to the tool context's state so the calling
    agent can access it as the definitive result. It also sets the action to
    skip LLM summarization of the tool's output.

    Args:
        tool_context: The context of the tool call.
        error_message: The error message string to save.

    Returns:
        The same error_message that was passed in.
    """
    if tool_context:
        tool_context.state[DOC_LINK_STATE_KEY] = error_message
        tool_context.actions.skip_summarization = True

    return error_message


async def _generate_with_marp_and_upload(
    markdown_content: str,
    output_filename_base: str,  # Base name without extension
    marp_format_flag: str,  # --pdf, --html, --pptx
    gcs_content_type: str,
    file_extension: str,  # .pdf, .html, .pptx
    tool_context: Optional[ToolContext] = None,
) -> str:
    """
    A generic internal function to generate a document with Marp and upload it.

    Args:
        markdown_content: The Marp-compatible Markdown string.
        output_filename_base: The base name for the output file (e.g., 'adk_report').
        marp_format_flag: The Marp CLI flag for the desired format (e.g., '--pdf').
        gcs_content_type: The MIME type for the GCS upload.
        file_extension: The file extension for the output file (e.g., '.pdf').
        tool_context: The ADK tool context.

    Returns:
        A string containing the result (a signed URL or an error message).
    """
    if not _check_marp_cli():
        error_message = (
            "Error: marp-cli is not installed or accessible. Cannot generate document."
        )
        return _handle_marp_error_in_context(tool_context, error_message)

    # Create a temporary directory for the output file.
    local_marp_output_dir = os.path.join(tempfile.gettempdir(), GENERATED_DOCS_SUBDIR)
    os.makedirs(local_marp_output_dir, exist_ok=True)

    # Sanitize the filename to prevent directory traversal or invalid characters.
    safe_local_filename = "".join(
        c if c.isalnum() or c in (".", "_", "-") else "_" for c in output_filename_base
    )
    local_output_path = os.path.join(
        local_marp_output_dir, safe_local_filename + file_extension
    )

    tmp_md_file_path = None
    try:
        # Write the markdown to a temporary file to be used as input for marp-cli.
        with tempfile.NamedTemporaryFile(
            mode="w+", delete=False, suffix=".md", encoding="utf-8"
        ) as tmp_md_file:
            tmp_md_file.write(markdown_content)
            tmp_md_file_path = tmp_md_file.name

        # Construct and run the marp-cli command as a subprocess.
        cmd = [
            MARP_CLI_COMMAND,
            tmp_md_file_path,
            marp_format_flag,
            "-o",
            local_output_path,
            "--allow-local-files",
        ]
        logger.info(f"Executing Marp command: {' '.join(cmd)}")
        result = subprocess.run(
            cmd, check=False, capture_output=True, text=True, encoding="utf-8"
        )

        # Handle errors from the marp-cli process.
        if result.returncode != 0:
            error_detail = (
                result.stderr.strip() if result.stderr else result.stdout.strip()
            )
            if not error_detail:
                error_detail = "No output from marp-cli."
            log_message = f"Error running marp-cli for {marp_format_flag} (code {result.returncode}): {error_detail}"
            logger.error(log_message)

            # Provide a detailed error message including the original markdown for debugging.
            error_message_to_user = f"{MARP_GENERATION_ERROR_MESSAGE_PREFIX}\nMARP CLI Error (code {result.returncode}): {error_detail}\n\nOriginal Markdown Content:\n{markdown_content}"
            # The original implementation of _handle_marp_error_in_context only takes 2 arguments
            return _handle_marp_error_in_context(
                tool_context,
                error_message_to_user,
                markdown_content,
                safe_local_filename,
            )

        # If the process succeeded, verify the output file was created and has content.
        if os.path.exists(local_output_path) and os.path.getsize(local_output_path) > 0:
            logger.info(
                f"Marp generated file locally: {local_output_path}. Marp stdout: {result.stdout}"
            )
            # Upload the generated file to GCS.
            return await _upload_to_gcs_and_get_signed_url(
                local_file_path=local_output_path,
                gcs_object_name_base=output_filename_base,
                content_type=gcs_content_type,
                file_extension=file_extension,
                tool_context=tool_context,
                markdown_content_for_fallback=markdown_content,
            )
        else:
            # Handle cases where marp-cli runs without error but produces no file.
            error_detail = f"Output file '{local_output_path}' not found or is empty after Marp CLI execution."
            if result.stdout:
                error_detail += f"\nMarp stdout: {result.stdout.strip()}"
            if result.stderr:
                error_detail += f"\nMarp stderr: {result.stderr.strip()}"
            logger.error(f"Marp file generation problem: {error_detail}")

            error_message_to_user = f"{MARP_GENERATION_ERROR_MESSAGE_PREFIX}\nError detail: {error_detail}\n\nOriginal Markdown Content:\n{markdown_content}"
            # The original implementation of _handle_marp_error_in_context only takes 2 arguments
            return _handle_marp_error_in_context(
                tool_context,
                error_message_to_user,
                markdown_content,
                safe_local_filename,
            )

    except FileNotFoundError:  # If the 'marp' command itself is not found.
        error_detail = f"'{MARP_CLI_COMMAND}' command not found. Please ensure marp-cli is installed and in PATH."
        logger.error(error_detail, exc_info=True)
        error_message_to_user = f"{MARP_GENERATION_ERROR_MESSAGE_PREFIX}\nError detail: {error_detail}\n\nOriginal Markdown Content:\n{markdown_content}"
        # The original implementation of _handle_marp_error_in_context only takes 2 arguments
        return _handle_marp_error_in_context(
            tool_context, error_message_to_user, markdown_content, safe_local_filename
        )
    except Exception as e:
        logger.error(
            f"An unexpected error occurred during {marp_format_flag} generation: {str(e)}",
            exc_info=True,
        )
        error_message_to_user = f"{MARP_GENERATION_ERROR_MESSAGE_PREFIX}\nError detail: An unexpected error occurred: {str(e)}\n\nOriginal Markdown Content:\n{markdown_content}"
        # The original implementation of _handle_marp_error_in_context only takes 2 arguments
        return _handle_marp_error_in_context(
            tool_context, error_message_to_user, markdown_content, safe_local_filename
        )
    finally:
        # Clean up temporary files.
        if tmp_md_file_path and os.path.exists(tmp_md_file_path):
            os.remove(tmp_md_file_path)
        if "local_output_path" in locals() and os.path.exists(local_output_path):
            try:
                # Check if the result variable exists and indicates Marp success.
                # If Marp was successful and the file is good, _upload_to_gcs_and_get_signed_url is responsible.
                # Otherwise, it's an error path, and we should clean up.
                marp_succeeded_and_file_good = (
                    "result" in locals()
                    and result.returncode == 0
                    and os.path.exists(local_output_path)
                    and os.path.getsize(local_output_path) > 0
                )

                if not marp_succeeded_and_file_good:
                    os.remove(local_output_path)
                # If marp_succeeded_and_file_good is true, _upload_to_gcs_and_get_signed_url
                # will use local_output_path and is responsible for its cleanup.
            except Exception as e_rem:
                logger.warning(
                    f"Could not remove temporary local Marp file {local_output_path}: {e_rem}"
                )


# --- Public-facing Functions for the Document Generator Agent's Tools ---


async def generate_pdf_from_markdown_with_gcs(
    markdown_content: str, output_filename: str, tool_context: ToolContext
) -> str:
    """Generates a PDF from Markdown and uploads it to GCS."""
    return await _generate_with_marp_and_upload(
        markdown_content,
        output_filename,
        "--pdf",
        "application/pdf",
        ".pdf",
        tool_context,
    )


async def generate_html_slides_from_markdown_with_gcs(
    markdown_content: str, output_filename: str, tool_context: ToolContext
) -> str:
    """Generates HTML slides from Markdown and uploads them to GCS."""
    return await _generate_with_marp_and_upload(
        markdown_content, output_filename, "--html", "text/html", ".html", tool_context
    )


async def generate_pptx_slides_from_markdown_with_gcs(
    markdown_content: str, output_filename: str, tool_context: ToolContext
) -> str:
    """Generates a PPTX presentation from Markdown and uploads it to GCS."""
    return await _generate_with_marp_and_upload(
        markdown_content,
        output_filename,
        "--pptx",
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        ".pptx",
        tool_context,
    )
