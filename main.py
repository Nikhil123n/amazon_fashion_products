from fastapi import FastAPI, HTTPException
from typing import List
from model_utils import find_similar_products, prepare_system

# Here we initialize FastAPI app
app = FastAPI(title="Amazon Fashion Product Similarity API")

# Initialize models and data once at startup
prepare_system()

# Route to get predictions for similar products 
@app.get("/find_similar_products", response_model=List[str])
def get_similar_products(product_id: str, num_similar: int) -> List[str]:
    try:
        similar_products = find_similar_products(product_id, num_similar)
        return similar_products
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error))
    except Exception as e:
        raise HTTPException(status_code=500, detail="Internal server error")

# For Local run and useful for debugging
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)