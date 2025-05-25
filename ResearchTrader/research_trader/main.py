"""
ResearchTrader FastAPI Application
Main entry point for the API
"""

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.responses import JSONResponse

from research_trader.config import settings

# Import the new and updated routers
from research_trader.router.papers import router as papers_router
from research_trader.router.qa import router as qa_router
from research_trader.router.strategy import router as strategy_router

# Removed imports for old search and summarize routers
from research_trader.utils.errors import ServiceError

# --- Logging Configuration --- >
log_level = settings.LOG_LEVEL.upper()
logging.basicConfig(
    level=log_level,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
logger.info(f"Logging configured with level: {log_level}")
# <---------------------------

# Create the FastAPI application
app = FastAPI(
    title="ResearchTrader API",
    description="API for discovering, exploring, and operationalizing ArXiv quantitative finance research",
    version="0.0.1",
    docs_url=None,
    redoc_url="/docs",
)

logger.info("Adding CORS middleware")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development/simplicity
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
logger.info("Including API routers")
app.include_router(papers_router)  # Includes prefix /papers and tag Papers
app.include_router(qa_router)  # Includes prefix /qa and tag Q&A
app.include_router(strategy_router)  # Includes prefix /strategy and tag Strategy Generation


# Add a simple root endpoint
@app.get("/", tags=["Root"])
def read_root():
    logger.debug("Root endpoint accessed")
    return {"message": "Welcome to ResearchTrader API v0.2.0"}


# Custom exception handler for ServiceError
@app.exception_handler(ServiceError)
async def service_error_handler(request: Request, exc: ServiceError):
    logger.warning(
        f"ServiceError handled: {exc.status_code} - {exc.detail} (Service: {exc.service})"
    )
    return JSONResponse(
        status_code=exc.status_code, content={"detail": exc.detail, "service": exc.service}
    )


# Custom exception handler for generic Exceptions (catch-all)
@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc.__class__.__name__}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500, content={"detail": "An unexpected internal server error occurred."}
    )


# Custom Swagger UI with better styling
@app.get("/swagger", include_in_schema=False)
async def custom_swagger_ui_html():
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - API Documentation",
        swagger_js_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js",
        swagger_css_url="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css",
    )


logger.info("FastAPI application initialized")

if __name__ == "__main__":
    import uvicorn

    logger.info("Starting Uvicorn server...")
    uvicorn.run(
        "research_trader.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=settings.API_RELOAD,
        log_level=settings.LOG_LEVEL.lower(),
    )
