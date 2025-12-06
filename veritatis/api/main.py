from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="Veritatis API", version="1.0")

# --- CORS setup ---
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # You can restrict this to your frontend later
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# --- Health check endpoint ---
@app.get("/health")
async def health_check():
   return {"status": "ok"}

# --- Error handler example ---
@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
   return JSONResponse(
      status_code=500,
      content={"message": f"Unexpected error: {exc}"}
   )
