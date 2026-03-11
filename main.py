import os
import torch  # Must be imported before other CUDA-enabled libs like Paddle
import sys
import pkgutil
import uvicorn
import nest_asyncio
import importlib
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
from app.config.config import Config

nest_asyncio.apply()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Handle startup/shutdown lifecycle with diagnostics."""
    print("Initializing all projects...")

    # Load environment & validate config
    Config.load_from_env_file()
    Config.validate()

    yield

    print("[INFO] Shutting down all services...")

# Initialize FastAPI
app = FastAPI(
    title="Aarvi Multi-Project AI Server",
    description="Server hosting multiple independent AI projects (cv_extract, rag_chatbot, etc.)",
    version="1.1.0",
    lifespan=lifespan
)

# Allow full CORS for dev
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create static directory if it doesn't exist
static_dir = os.path.join(os.getcwd(), "static")
if not os.path.exists(static_dir):
    os.makedirs(static_dir)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

def include_project_routes():
    """Dynamically import all project routers with safe fallbacks."""
    import app.projects as projects_pkg
    from app.projects.unified.routes import unified_routes # Ensure new project is detectable

    print("[PROJECTS] Searching for projects in:", projects_pkg.__path__)
    
    # Explicitly include unified project to ensure it works
    try:
        from app.projects.unified.routes import unified_routes
        app.include_router(unified_routes.router, prefix="/unified", tags=["UNIFIED"])
        print("[SUCCESS] Explicitly mounted unified routes")
    except Exception as e:
        print(f"[ERROR] Failed to mount unified routes: {e}")

    found_projects = True # Mark as true since we mounted unified

    for _, project_name, ispkg in pkgutil.iter_modules(projects_pkg.__path__):
        if not ispkg:
            continue

        routes_dir = os.path.join("app", "projects", project_name, "routes")
        if not os.path.isdir(routes_dir):
            print(f"[WARNING] No routes folder for project '{project_name}' — skipping.")
            continue

        # Auto-detect any *_routes.py file
        loaded = False
        for file in os.listdir(routes_dir):
            if file.endswith("_routes.py"):
                module_name = file[:-3]
                module_path = f"app.projects.{project_name}.routes.{module_name}"
                try:
                    print(f"[LOAD] Importing routes from: {module_path}")
                    router_module = importlib.import_module(module_path)
                    router = getattr(router_module, "router", None)
                    if router:
                        app.include_router(router, prefix=f"/{project_name}", tags=[project_name.upper()])
                        print(f"[SUCCESS] Mounted routes for project: {project_name} ({module_name})")
                        found_projects = True
                        loaded = True
                        break
                    else:
                        print(f"[WARNING] Router variable not found in {module_name}.py")
                except ModuleNotFoundError as e:
                    print(f"[ERROR] Skipped '{project_name}' — Missing dependency: {e.name}")
                except Exception as e:
                    print(f"[ERROR] Error loading '{project_name}': {e}")
        if not loaded:
            print(f"[WARNING] No valid route file found for '{project_name}'.")

    if not found_projects:
        print("[WARNING] No project routes mounted. Check folder structure or dependencies.")

# Load all routes dynamically
include_project_routes()

@app.get("/")
def root():
    """Root endpoint listing all available APIs."""
    active_routes = [
        {"path": route.path, "name": route.name}
        for route in app.routes
        if route.path not in ("/", "/openapi.json", "/docs")
    ]
    return {
        "server": "Aarvi Multi-Project AI Server",
        "projects": active_routes,
        "status": "running"
    }

def run_server():
    print("Starting Aarvi Multi-Project AI Server")
    print(f"[INFO] Running at: http://{Config.HOST}:{Config.PORT}")
    uvicorn.run("app.main:app", host=Config.HOST, port=Config.PORT)

if __name__ == "__main__":
    run_server()
