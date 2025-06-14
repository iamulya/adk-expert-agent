"""
This module provides utility functions for document generation using Marp CLI
and for interacting with Google Cloud Storage (GCS).

These helpers are used by the document generation tools to handle file I/O,
GCS uploads, and signed URL creation, encapsulating the common logic.
"""

import os
import logging
import subprocess
import uuid
import datetime
from typing import Optional

from google.cloud import storage
from google.adk.sessions.state import State
from google.adk.tools import ToolContext
from google.genai import types as genai_types

from ..config import (
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL,
    SIGNED_URL_EXPIRATION_SECONDS,
)

logger = logging.getLogger(__name__)

# --- Constants ---
MARP_CLI_COMMAND = "marp"
GENERATED_DOCS_SUBDIR = "generated_documents_from_adk_agent"
# A state key used by tools to pass the final document URL back to the calling agent.
DOC_LINK_STATE_KEY = State.TEMP_PREFIX + "generated_document_signed_url"


def _check_marp_cli():
    """Checks if the marp-cli command is installed and accessible in the system's PATH."""
    try:
        subprocess.run([MARP_CLI_COMMAND, "--version"], check=True, capture_output=True)
        logger.info(f"'{MARP_CLI_COMMAND}' found and working.")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        logger.error(f"ERROR: '{MARP_CLI_COMMAND}' not found or not executable.")
        return False


async def _upload_to_gcs_and_get_signed_url(
    local_file_path: str,
    gcs_object_name_base: str,
    content_type: str,
    file_extension: str,
    tool_context: Optional[ToolContext] = None,
    markdown_content_for_fallback: Optional[str] = None,
) -> str:
    """
    Uploads a local file to GCS and generates a v4 signed URL for it.

    If GCS is not configured, or if the upload fails, it attempts to save the
    original Markdown content as a fallback ADK artifact.

    Args:
        local_file_path: The path to the locally generated file to upload.
        gcs_object_name_base: The base name for the GCS object.
        content_type: The MIME content type of the file.
        file_extension: The file extension (e.g., '.pdf').
        tool_context: The ADK tool context for state management and artifacts.
        markdown_content_for_fallback: The source Markdown to save if upload fails.

    Returns:
        A string containing the success message with the signed URL, or an error message.
    """
    session_id_for_path = (
        tool_context._invocation_context.session.id
        if tool_context and hasattr(tool_context, "_invocation_context")
        else "unknown_session"
    )
    unique_id = uuid.uuid4().hex[:8]

    gcs_object_name_base_cleaned = gcs_object_name_base.strip("/").replace(
        file_extension, ""
    )
    gcs_object_name = f"adk_generated_documents/{session_id_for_path}/{gcs_object_name_base_cleaned}_{unique_id}{file_extension}"
    adk_artifact_filename = (
        f"{os.path.basename(gcs_object_name_base_cleaned)}_{unique_id}{file_extension}"
    )

    # Proceed only if GCS configuration is provided.
    if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
        try:
            logger.info(
                f"Attempting to save to GCS bucket '{GCS_BUCKET_NAME}' as object '{gcs_object_name}'"
            )
            storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
            bucket = storage_client.bucket(GCS_BUCKET_NAME)
            blob = bucket.blob(gcs_object_name)

            blob.upload_from_filename(local_file_path, content_type=content_type)
            logger.info(
                f"File successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}"
            )

            signed_url_str = "No signed URL generated if SA email is not set."
            # Generate a signed URL only if the service account for impersonation is configured.
            if GCS_SIGNED_URL_SA_EMAIL:
                from google import auth

                principal_credentials, _ = auth.default(
                    scopes=["<https://www.googleapis.com/auth/cloud-platform>"]
                )
                from google.auth import impersonated_credentials

                impersonated_target_credentials = impersonated_credentials.Credentials(
                    source_credentials=principal_credentials,
                    target_principal=GCS_SIGNED_URL_SA_EMAIL,
                    target_scopes=[
                        "<https://www.googleapis.com/auth/devstorage.read_write>"
                    ],
                    lifetime=SIGNED_URL_EXPIRATION_SECONDS + 60,
                )

                signed_url_str = blob.generate_signed_url(
                    version="v4",
                    expiration=datetime.timedelta(
                        seconds=SIGNED_URL_EXPIRATION_SECONDS
                    ),
                    method="GET",
                    credentials=impersonated_target_credentials,
                )
                logger.info(f"Generated GCS signed URL: {signed_url_str}")
                output = f"Document generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url_str}"
                # Store the final success message in the tool context state.
                if tool_context:
                    tool_context.state[DOC_LINK_STATE_KEY] = output
                    logger.info(
                        f"Saved signed URL to tool context state under key: {DOC_LINK_STATE_KEY}"
                    )
                    tool_context.actions.skip_summarization = True
                return output
            else:
                logger.warning(
                    "GCS_SIGNED_URL_SA_EMAIL not configured. Returning public GCS path instead of signed URL."
                )
                public_url = f"<https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}>"
                return f"Document generated and saved to Google Cloud Storage at: gs://{GCS_BUCKET_NAME}/{gcs_object_name} (Public URL if bucket/object is public: {public_url}). Signed URL generation skipped as service account for signing is not configured."

        except Exception as e:
            logger.error(
                f"Error during GCS storage or signed URL generation for {gcs_object_name}: {e}",
                exc_info=True,
            )
            # If GCS upload fails, save the source markdown as a fallback ADK artifact for debugging.
            if tool_context and markdown_content_for_fallback:
                logger.warning("Falling back to ADK in-memory artifact service.")
                await tool_context.save_artifact(
                    adk_artifact_filename + ".md",
                    genai_types.Part(text=markdown_content_for_fallback),
                )
                return f"Error saving to GCS. Source Markdown saved to ADK artifacts as: {adk_artifact_filename}.md. GCS Error: {str(e)}"
            return f"Error saving to GCS: {str(e)}. Fallback to ADK artifact not possible without tool_context/markdown."
    else:
        # Fallback if GCS is not configured at all.
        if tool_context and markdown_content_for_fallback:
            logger.warning(
                "GCS bucket/project not configured. Saving source Markdown to ADK in-memory artifact service."
            )
            await tool_context.save_artifact(
                adk_artifact_filename + ".md",
                genai_types.Part(text=markdown_content_for_fallback),
            )
            return f"GCS not configured. Source Markdown for {adk_artifact_filename} saved as ADK artifact: {adk_artifact_filename}.md."
        logger.warning(
            "GCS bucket/project not configured. Cannot save artifact as tool_context or markdown_content is missing."
        )
        return "GCS not configured and could not save to ADK artifacts."
