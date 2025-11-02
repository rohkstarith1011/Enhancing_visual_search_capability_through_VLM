# File: server.py
import sys
import time
import uvicorn
from pyngrok import ngrok
import nest_asyncio
import logging
import os
from main import app # <-- IMPORT the FastAPI app instance from main.py

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Ngrok Authentication ---
# ⚠️ WARNING: Avoid hardcoding tokens in production code. Use environment variables.
NGROK_AUTH_TOKEN = "34ay3vLPYfk5iBjA7kCkAcLGYyh_66Hzs6xLWK8No4XKy1WYu" # <-- PASTE YOUR TOKEN HERE

# --- Server Configuration ---
HOST = "0.0.0.0" # Listen on all available network interfaces
PORT = 5000     # Port the server will run on locally

# ==============================================================================
# Main Execution Block (Setup ngrok and Run Uvicorn)
# ==============================================================================
if __name__ == "__main__":
    public_url = None
    logger.info("--- Setting up ngrok tunnel ---")

    if NGROK_AUTH_TOKEN == "YOUR_NGROK_AUTH_TOKEN" or not NGROK_AUTH_TOKEN:
        logger.error("❌ ERROR: ngrok Auth Token is not set! Get one from dashboard.ngrok.com")
    else:
        try:
            ngrok.kill() # Kill existing tunnels
            ngrok.set_auth_token(NGROK_AUTH_TOKEN)
            nest_asyncio.apply() # Patch for Colab/Jupyter compatibility
            logger.info("Connecting to ngrok...") # Log before connect
            http_tunnel = ngrok.connect(PORT) # Tunnel to the configured PORT
            public_url = http_tunnel.public_url
            logger.info(f"✅ ngrok tunnel created successfully!")

            # --- Explicitly PRINT and FLUSH the URL ---
            print("\n" + "="*50)
            print(f"👉 FastAPI backend accessible at: {public_url}")
            print("="*50 + "\n")
            sys.stdout.flush() # Force print output immediately
            # --- ---

            # Optional: Add a small delay to ensure URL is visible
            logger.info("Starting server in 3 seconds...")
            time.sleep(3)

        except Exception as e:
            logger.error(f"❌ Error creating ngrok tunnel: {e}")
            logger.error("   Ensure token is correct and ngrok is installed/reachable.")

    if public_url:
        logger.info("\n--- Starting Uvicorn server for FastAPI ---")
        try:
            # Tell Uvicorn to run the 'app' object imported from 'main'
            uvicorn.run("main:app", host=HOST, port=PORT, log_level="info") # Pass app as string "module:variable"
        except KeyboardInterrupt:
            logger.info("Server stopped by user (KeyboardInterrupt).")
        except SystemExit as e:
            logger.info(f"Server stopped (SystemExit: {e}).")
        except Exception as e:
            logger.exception(f"❌ Error running Uvicorn server: {e}")
            logger.error("   Check logs for errors during FastAPI startup (resource loading in main.py).")
        finally:
            logger.info("Shutting down ngrok tunnel.")
            ngrok.kill() # Clean up ngrok tunnel when server stops
    else:
        logger.error("\n❌ Skipping Uvicorn server start because ngrok tunnel failed.")