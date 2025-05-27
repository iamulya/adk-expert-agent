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
import google.auth.exceptions # No longer need google.oauth2.service_account or google.auth.transport.requests

from google.adk.sessions.state import State
from google.adk.tools import BaseTool, ToolContext
import google.genai.types as genai_types

from ..config import ( # Relative import
    GCS_BUCKET_NAME,
    GCS_PROJECT_ID_FOR_BUCKET,
    GCS_SIGNED_URL_SA_EMAIL,
    SIGNED_URL_EXPIRATION_SECONDS,
    MERMAID_CLI_PATH
)

logger = logging.getLogger(__name__)
GCS_LINK_STATE_KEY = State.TEMP_PREFIX + "gcs_link_for_diagram"

PUPPETEER_CONFIG_PATH = os.getenv("PUPPETEER_CONFIG_PATH", "/app/puppeteer-config.json")

class MermaidToPngAndUploadTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="mermaid_to_png_and_gcs_upload",
            description="Converts Mermaid diagram syntax to a PNG image, uploads it to Google Cloud Storage, and returns a signed URL for access.",
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
                        description="The Mermaid diagram syntax to convert, enclosed in ```mermaid ... ```.",
                    )
                },
                required=["mermaid_syntax"],
            ),
        )

    async def run_async(self, args: Dict[str, str], tool_context: ToolContext) -> str:
        mermaid_syntax_raw = args.get("mermaid_syntax")
        if not mermaid_syntax_raw:
            return "Error: Mermaid syntax is required."

        match = re.search(r"```mermaid\s*([\s\S]+?)\s*```", mermaid_syntax_raw, re.DOTALL)
        if match:
            mermaid_syntax = match.group(1).strip()
        else:
            mermaid_syntax = mermaid_syntax_raw.strip()

        if not mermaid_syntax:
            return "Error: Extracted Mermaid syntax is empty."

        session_id = tool_context._invocation_context.session.id if tool_context and hasattr(tool_context, '_invocation_context') and hasattr(tool_context._invocation_context, 'session') else "unknown_session"
        
        png_data = None
        mmd_file_path = None
        png_file_path = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".mmd", encoding='utf-8') as mmd_file:
                mmd_file.write(mermaid_syntax)
                mmd_file_path = mmd_file.name

            with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as png_temp_obj:
                png_file_path = png_temp_obj.name

            logger.info(f"Running mmdc: {MERMAID_CLI_PATH} -p {PUPPETEER_CONFIG_PATH} -i {mmd_file_path} -o {png_file_path}")
            process = await asyncio.create_subprocess_exec(
                MERMAID_CLI_PATH,
                "-p", PUPPETEER_CONFIG_PATH,
                "-i", mmd_file_path,
                "-o", png_file_path,
                stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                error_msg = f"Mermaid CLI error (code {process.returncode}): {stderr.decode() if stderr else stdout.decode()}"
                logger.error(error_msg)
                if tool_context:
                    await tool_context.save_artifact(
                        f"error_mermaid_syntax_{uuid.uuid4().hex[:8]}.mmd",
                        genai_types.Part(text=mermaid_syntax)
                    )
                return f"Error converting Mermaid to PNG: {error_msg}. The original syntax was saved as an artifact."

            with open(png_file_path, "rb") as f:
                png_data = f.read()
            logger.info(f"Successfully converted Mermaid to PNG: {png_file_path}")

        except FileNotFoundError:
            logger.error(f"Mermaid CLI (mmdc) not found at '{MERMAID_CLI_PATH}'. Please install it or set MERMAID_CLI_PATH.")
            return f"Error: Mermaid CLI (mmdc) not found at '{MERMAID_CLI_PATH}'. Cannot generate PNG."
        except Exception as e:
            logger.error(f"Error during PNG generation: {e}", exc_info=True)
            return f"Error generating PNG: {str(e)}"
        finally:
            if mmd_file_path and os.path.exists(mmd_file_path):
                os.remove(mmd_file_path)
            if png_file_path and os.path.exists(png_file_path):
                os.remove(png_file_path)

        if not png_data:
            return "Error: PNG data could not be generated."

        gcs_object_name = f"mermaid_diagrams/{session_id}/diagram_{uuid.uuid4().hex[:8]}.png"
        adk_artifact_filename = f"mermaid_diagram_{uuid.uuid4().hex[:8]}.png"

        if GCS_BUCKET_NAME and GCS_PROJECT_ID_FOR_BUCKET:
            try:
                logger.info(f"Attempting to save PNG to GCS bucket '{GCS_BUCKET_NAME}' in project '{GCS_PROJECT_ID_FOR_BUCKET}' as object '{gcs_object_name}'")
                storage_client = storage.Client(project=GCS_PROJECT_ID_FOR_BUCKET)
                bucket = storage_client.bucket(GCS_BUCKET_NAME)
                blob = bucket.blob(gcs_object_name)
                blob.upload_from_string(png_data, content_type='image/png')
                logger.info(f"PNG successfully uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}")

                if not GCS_SIGNED_URL_SA_EMAIL:
                    logger.warning("GCS_SIGNED_URL_SA_EMAIL not set. Cannot generate signed URL.")
                    public_url = f"https://storage.googleapis.com/{GCS_BUCKET_NAME}/{gcs_object_name}"
                    return f"Diagram PNG uploaded to GCS: gs://{GCS_BUCKET_NAME}/{gcs_object_name}. Public URL (if public): {public_url}. Signed URL not generated as SA email is missing."

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
                            lifetime=120
                        )
                    signed_url_str = blob.generate_signed_url(
                        version="v4",
                        expiration=datetime.timedelta(seconds=SIGNED_URL_EXPIRATION_SECONDS),
                        method="GET",
                        credentials=impersonated_target_credentials,
                    )
                    logger.info(f"Generated signed URL: {signed_url_str}")
                    output = f"Diagram PNG generated and saved to Google Cloud Storage. Download (link expires in {SIGNED_URL_EXPIRATION_SECONDS // 60} mins): {signed_url_str}"
                    if tool_context:
                        tool_context.state[GCS_LINK_STATE_KEY] = output
                        logger.info(f"Saved signed URL to tool context state under key: {GCS_LINK_STATE_KEY}")
                        tool_context.actions.skip_summarization = True
                    return output
                except google.auth.exceptions.RefreshError as refresh_err:
                    logger.error(f"Error refreshing impersonated credentials for Mermaid: {refresh_err}. Check SA '{GCS_SIGNED_URL_SA_EMAIL}'.", exc_info=True)
                    if tool_context:
                        await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                    return f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(refresh_err)}"
                except Exception as e_sign:
                    logger.error(f"Error during signed URL generation for Mermaid: {e_sign}", exc_info=True)
                    if tool_context:
                        await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                    return f"Error generating signed URL for Mermaid diagram. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_sign)}"
            except Exception as e_gcs:
                logger.error(f"Error during GCS PNG storage for Mermaid: {e_gcs}", exc_info=True)
                if tool_context:
                    await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
                return f"Error saving Mermaid diagram to GCS. PNG file saved to ADK artifacts as: {adk_artifact_filename}. GCS Error: {str(e_gcs)}"
        else:
            logger.info("GCS bucket/project for Mermaid not configured. Saving PNG to ADK in-memory artifact service.")
            if tool_context:
                await tool_context.save_artifact(adk_artifact_filename, genai_types.Part(inline_data=genai_types.Blob(mime_type="image/png", data=png_data)))
            return f"Mermaid diagram PNG generated and saved as ADK artifact: {adk_artifact_filename}."

mermaid_gcs_tool_instance = MermaidToPngAndUploadTool()
