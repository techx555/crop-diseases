import json
import os
from dotenv import load_dotenv
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from google import genai
from google.genai import errors, types
from pydantic import BaseModel

# 1. Load environment variables from .env
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is missing. Please set it in your .env file.")

# Initialize the Google GenAI client
client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(
    title="Crop Disease Identification API",
    description="Backend API for detecting crop diseases from plant imagery using Gemini 2.5 Flash",
    version="1.0.0",
)

# Configure CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}

# 2. Define strict Pydantic schema for Gemini's JSON output
class CropAnalysisResponse(BaseModel):
    is_plant: bool
    crop_name: str
    disease_name: str
    confidence_estimate: str
    summary: str
    symptoms: list[str]
    causes: list[str]
    treatment_and_prevention: list[str]


# 3. Strict system prompt for the AI agricultural pathologist
SYSTEM_PROMPT = """
You are an expert agricultural pathologist and agronomist. Analyze the uploaded image and return a structured JSON diagnosis.

Follow these strict rules:
1. If the image does NOT contain a plant, leaf, crop, or agricultural specimen, set "is_plant" to false, "crop_name" to "Not a plant", "disease_name" to "N/A", "confidence_estimate" to "N/A", explain what is shown in "summary", and return empty lists [] for "symptoms", "causes", and "treatment_and_prevention".
2. If the image is too blurry, dark, or unclear to make a reliable diagnosis, set "is_plant" to true (if a plant is visible), "crop_name" to "Unclear", "disease_name" to "Unclear Image", "confidence_estimate" to "Low (0%)", explain in "summary" why a clearer photo is needed, and provide tips for taking a better photo in "treatment_and_prevention".
3. If the plant appears completely healthy, set "is_plant" to true, identify the "crop_name", set "disease_name" to "Healthy", provide a "confidence_estimate" (e.g., "High (92%)"), write a detailed paragraph in "summary" describing the healthy foliage, return empty lists [] for "symptoms" and "causes", and list general maintenance best practices in "treatment_and_prevention".
4. If a disease, pest, or nutrient deficiency is detected, set "is_plant" to true, identify the "crop_name" and specific "disease_name", provide a "confidence_estimate" (e.g., "High (85%)" or "Medium (65%)"), write a detailed paragraph in "summary" explaining the condition, and populate "symptoms", "causes", and "treatment_and_prevention" with clear, actionable steps for a farmer.
"""


@app.get("/")
@app.get("/health")
async def health_check():
    return {
        "status": "online",
        "message": "Crop Disease API is running",
    }


@app.post("/api/analyze-crop", response_model=CropAnalysisResponse)
async def analyze_crop(file: UploadFile = File(...)):
    # Validate uploaded file MIME type
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail="Invalid file type. Only JPEG, PNG, and WEBP images are allowed.",
        )

    # Read image bytes and verify non-empty payload
    file_bytes = await file.read()
    if not file_bytes:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image file is empty.",
        )

    try:
        # Call Gemini 2.5 Flash with image bytes and strict JSON schema
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                types.Part.from_bytes(
                    data=file_bytes,
                    mime_type=file.content_type,
                ),
                "Inspect this image and provide a complete agricultural pathology analysis.",
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_PROMPT,
                response_mime_type="application/json",
                response_schema=CropAnalysisResponse,
                temperature=0.2,
            ),
        )

        # Ensure the model returned text content
        if not response.text:
            raise HTTPException(
                status_code=422,
                detail="The AI could not interpret the uploaded image. Please upload a clearer photo of the plant leaf or crop.",
            )

        # Parse and return the validated JSON dictionary
        analysis_data = json.loads(response.text)
        return analysis_data

    except errors.APIError as api_err:
        # Handle Gemini API errors (rate limits, invalid key, service overload)
        raise HTTPException(
            status_code=502,
            detail=f"Gemini API error: {getattr(api_err, 'message', str(api_err))}",
        )
    except json.JSONDecodeError:
        # Handle unexpected malformed JSON from the model
        raise HTTPException(
            status_code=500,
            detail="Failed to parse the AI response into valid JSON.",
        )
    except HTTPException:
        # Re-raise explicit FastAPI HTTPExceptions untouched
        raise
    except Exception as exc:
        # Catch-all for unexpected runtime or network errors
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred during crop analysis: {str(exc)}",
        )