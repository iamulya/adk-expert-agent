# expert-agents/tools/mermaid_to_png_and_upload_tool.py
import asyncio
import datetime
import logging
import os
import re
import subprocess
import tempfile
import uuid
from typing import Dict

from google.cloud import storage
import google.auth.exceptions

from google.adk.sessions.state import State
from google.adk.tools import BaseTool, ToolContext
import google.genai.types as genai_types

from ..config import (
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL,
    SIGNED_URL_EXPIRATION_SECONDS,
    MERMAID_CLI_PATH
)

logger = logging.getLogger(__name__)
GCS_LINK_STATE_KEY = State.TEMP_PREFIX + "gcs_link_for_diagram"
# This prefix helps the calling agent understand the nature of the error if needed,
# but the primary goal is that the tool itself returns a comprehensive error message.
PNG_CREATION_ERROR_MESSAGE_PREFIX = "There was an error during the creation of the png file."

PUPPETEER_CONFIG_PATH = os.getenv("PUPPETEER_CONFIG_PATH", "/app/puppeteer-config.json")

class MermaidToPngAndUploadTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="mermaid_to_png_and_gcs_upload",
            description="Converts Mermaid diagram syntax to a PNG image, uploads it to Google Cloud Storage, and returns a signed URL for access. If PNG creation fails, it returns an error message including the original syntax.",
        )

    def _get_declaration(self):
        return genai_types.FunctionDeclaration(
            name=self.name,
            description=self.description,
            parameters=genai_types.Schema(
                type=genai_types.Type.OBJECT,
                properties={
                    "mermaid_syntax": genai_types.Schema(
                        type=genai_types.Type.STRING,
                        description="The Mermaid diagram syntax to convert, enclosed in ```mermaid ... ``` or as raw syntax.",
                    )
                },
                required=["mermaid_syntax"],
            ),
        )

    async def run_async(self, args: Dict[str, str], tool_context: ToolContext) -> str:
        tool_context.actions.skip_summarization = True # Tool provides the final message
        final_output_message = ""

        mermaid_syntax_raw_input = args.get("mermaid_syntax")
        if not mermaid_syntax_raw_input:
            final_output_message = "Error: Mermaid syntax is required."
            if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
            return final_output_message

        match = re.search(r"```mermaid\s*([\s\S]+?)\s*```", mermaid_syntax_raw_input, re.DOTALL)
        if match:
            mermaid_syntax_for_conversion = match.group(1).strip()
        else:
            mermaid_syntax_for_conversion = mermaid_syntax_raw_input.strip()

        if not mermaid_syntax_for_conversion:
            final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\nExtracted Mermaid syntax is empty.\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
            if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
            return final_output_message

        session_id = tool_context._invocation_context.session.id if tool_context and hasattr(tool_context, '_invocation_context') and hasattr(tool_context._invocation_context, 'session') else "unknown_session"
        
        png_data = None
        mmd_file_path = None
        png_file_path_local_temp = None # Use a different name to avoid conflict with GCS path
        mermaid_syntax_for_conversion = mermaid_syntax_for_conversion.replace("```", "").replace("mermaid", "\n") #provide only the mermaid syntax without the code block markers

        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".mmd", encoding='utf-8') as mmd_file:
                mmd_file.write(mermaid_syntax_for_conversion)
                mmd_file_path = mmd_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as png_temp_obj:
                png_file_path_local_temp = png_temp_obj.name

            logger.info(f"Running mmdc: {MERMAID_CLI_PATH} -p {PUPPETEER_CONFIG_PATH} -i {mmd_file_path} -o {png_file_path_local_temp}")
            process = await asyncio.create_subprocess_exec(
                MERMAID_CLI_PATH,
                "-p", PUPPETEER_CONFIG_PATH,
                "-i", mmd_file_path,
                "-o", png_file_path_local_temp,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg_detail = stderr.decode().strip() if stderr else stdout.decode().strip()
                error_msg = f"Mermaid CLI error (code {process.returncode}): {error_msg_detail}"
                logger.error(error_msg)
                if tool_context:
                    await tool_context.save_artifact(
                        f"error_mermaid_syntax_{uuid.uuid4().hex[:8]}.mmd",
                        genai_types.Part(text=mermaid_syntax_raw_input)
                    )
                final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\n{error_msg}\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
                if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
                return final_output_message

            if not os.path.exists(png_file_path_local_temp) or os.path.getsize(png_file_path_local_temp) == 0:
                error_msg = "Mermaid CLI ran but PNG file was not created or is empty."
                logger.error(error_msg)
                if tool_context:
                     await tool_context.save_artifact(
                        f"error_mermaid_syntax_empty_png_{uuid.uuid4().hex[:8]}.mmd",
                        genai_types.Part(text=mermaid_syntax_raw_input)
                    )
                final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\n{error_msg}\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
                if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
                return final_output_message
            
            with open(png_file_path_local_temp, "rb") as f:
                png_data = f.read()
            logger.info(f"Successfully converted Mermaid to PNG: {png_file_path_local_temp}")

        except FileNotFoundError:
            error_msg = f"Mermaid CLI (mmdc) not found at '{MERMAID_CLI_PATH}'. Please install it or set MERMAID_CLI_PATH."
            logger.error(error_msg)
            final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\n{error_msg}\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
            if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
            return final_output_message
        except Exception as e:
            error_msg = f"An unexpected error occurred during PNG generation: {str(e)}"
            logger.error(error_msg, exc_info=True)
            if tool_context: 
                await tool_context.save_artifact(
                    f"unexpected_error_mermaid_syntax_{uuid.uuid4().hex[:8]}.mmd",
                    genai_types.Part(text=mermaid_syntax_raw_input)
                )
            final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\n{error_msg}\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
            if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
            return final_output_message
        finally:
            if mmd_file_path and os.path.exists(mmd_file_path):
                os.remove(mmd_file_path)

        if not png_data:
            error_msg = "PNG data could not be generated (file was empty or unreadable after CLI success)."
            logger.error(error_msg)
            final_output_message = f"{PNG_CREATION_ERROR_MESSAGE_PREFIX}\n{error_msg}\n\nOriginal Mermaid Syntax:\n{mermaid_syntax_raw_input}"
            if tool_context: tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
            return final_output_message

        gcs_object_name = f"mermaid_diagrams/{session_id}/diagram_{uuid.uuid4().hex[:8]}.png"
        adk_artifact_filename = f"mermaid_diagram_{uuid.uuid4().hex[:8]}.png"

        try:
            if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
                logger.info(f"Attempting to save PNG to GCS bucket '{GCS_BUCKET_NAME}' in project '{GCS_PROJECT_ID_FOR_BUCKET}' as object '{gcs_object_name}'")
                storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(gcs_object_name)
                
                blob.upload_from_filename(png_file_path_local_temp, content_type='image/png')
                logger.info(f"PNG successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}")

                if not GCS_SIGNED_URL_SA_EMAIL:
                    logger.warning("GCS_SIGNED_URL_SA_EMAIL not set. Cannot generate signed URL.")
                    public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}"
                    final_output_message = f"Diagram PNG uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}. Public URL (if public): {public_url}. Signed URL not generated as SA email is missing."
                else:
                    try:
                        from google import auth
                        principal_credentials, _ = auth.default(
                            scopes=['https://www.googleapis.com/auth/cloud-platform']
                        )
                        from google.auth import impersonated_credentials
                        impersonated_target_credentials = impersonated_credentials.Credentials(
                                source_credentials=principal_credentials,
                                target_principal=GCS_SIGNED_URL_SA_EMAIL,
                                target_scopes=['https://www.googleapis.com/auth/devstorage.read_write'],
                                lifetime=SIGNED_URL_EXPIRATION_SECONDS + 60 
                            )
                        signed_url_str = blob.generate_signed_url(
                            version="v4",
                            expiration=datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS),
                            method="GET",
                            credentials=impersonated_target_credentials,
                        )
                        logger.info(f"Generated signed URL: {signed_url_str}")
                        final_output_message = f"Diagram PNG generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url_str}"
                    
                    except google.auth.exceptions.RefreshError as refresh_err:
                        logger.error(f"Error refreshing impersonated credentials for Mermaid: {refresh_err}. Check SA '{GCS_SIGNED_URL_SA_EMAIL}'.", exc_info=True)
                        if tool_context:
                            await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                        final_output_message = f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(refresh_err)}"
                    except Exception as e_sign:
                        logger.error(f"Error during signed URL generation for Mermaid: {e_sign}", exc_info=True)
                        if tool_context:
                            await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                        final_output_message = f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_sign)}"
            else: 
                logger.info("GCS bucket/project for Mermaid not configured. Saving PNG to ADK in-memory artifact service.")
                if tool_context:
                    await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                final_output_message = f"Mermaid diagram PNG generated and saved as ADK artifact: {adk_artifact_filename}."
        
        except Exception as e_gcs_upload: 
            logger.error(f"Error during GCS PNG storage for Mermaid: {e_gcs_upload}", exc_info=True)
            if tool_context: # Save the successfully generated PNG as an artifact even if GCS upload fails
                await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
            final_output_message = f"Error saving Mermaid diagram to GCS. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_gcs_upload)}"
        finally:
            if png_file_path_local_temp and os.path.exists(png_file_path_local_temp):
                try:
                    os.remove(png_file_path_local_temp)
                except Exception as e_rem_png:
                    logger.warning(f"Could not remove temporary local PNG file {png_file_path_local_temp}: {e_rem_png}")
        
        if tool_context:
            tool_context.state[GCS_LINK_STATE_KEY] = final_output_message
        return final_output_message

mermaid_gcs_tool_instance = MermaidToPngAndUploadTool()