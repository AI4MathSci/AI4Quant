# QuantFinance

A web application designed to allow users to choose algorithms in quant finance areas for simualtion and training. 

## Setup and Running

### Prerequisites

- **Backend:** Python 3.10+, FastAPI, Pydantic
- **Frontend:** Gradio
- [uv](https://github.com/astral-sh/uv) - A fast Python package installer and resolver

### Method 1:
1. Clone the repository:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant
```
2. Create a virtual environment with uv:
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```
3. Install dependencies with uv:
```bash
uv pip install .
```
4. Run the backend, start the backend server:
```bash
python quantfin/backend/main.py 
```
5. Run the frontend, start the frontend server:
```bash
python quantfin/frontend/app.py 
```

### Method 2:
1. Clone the repository:
```bash
git clone https://github.com/AI4MathSci/AI4Quant.git
cd AI4Quant
```
2. Create a virtual environment and install backend dependencies with uv, and start backend server:
```bash
uv run python quantfin/backend/main.py
```
3. Install frontend dependencies (if any) with uv, and start frontend server:
```bash
uv run python quantfin/frontend/app.py 
```
The frontend will typically be available at `http://localhost:7860`, you can start a browser and open that URL to access the QuantFinacne system
