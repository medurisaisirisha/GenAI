from fastapi import FastAPI, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from io import BytesIO

app = FastAPI()

def preprocess_data(file_contents):
    data = pd.read_csv(BytesIO(file_contents))
    data['Date'] = pd.to_datetime(data['Date'])
    return data

@app.post("/calculate-gross-margin/")
async def calculate_gross_margin(file: UploadFile = UploadFile(...)):
    try:
        contents = await file.read()
        data = preprocess_data(contents)

        # Calculate gross margin within the endpoint
        data['Profit'] = data['Selling price'] - data['Buying price']
        gross_margin = (data['Profit'].sum() / data['Buying price'].sum()) * 100

        result = {"gross_margin": gross_margin}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/most-profitable-vendor/")
async def most_profitable_vendor(file: UploadFile = UploadFile(...)):
    try:
        contents = await file.read()
        data = preprocess_data(contents)

        # Calculate profit for each vendor within the endpoint
        data['Profit'] = data['Selling price'] - data['Buying price']

        # Calculate profit for each vendor
        vendor_profit = data.groupby('Firm bought from')['Profit'].sum()
        most_profitable_vendor = vendor_profit.idxmax()

        result = {"most_profitable_vendor": most_profitable_vendor}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/least-profitable-customer/")
async def least_profitable_customer(file: UploadFile = UploadFile(...)):
    try:
        contents = await file.read()
        data = preprocess_data(contents)

        # Calculate profit for each customer within the endpoint
        data['Profit'] = data['Selling price'] - data['Buying price']

        # Calculate profit for each customer
        customer_profit = data.groupby('Customer')['Profit'].sum()
        least_profitable_customer = customer_profit.idxmin()

        result = {"least_profitable_customer": least_profitable_customer}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/most-profitable-day/")
async def most_profitable_day(file: UploadFile = UploadFile(...)):
    try:
        contents = await file.read()
        data = preprocess_data(contents)

        # Calculate profit for each day of the week within the endpoint
        data['Profit'] = data['Selling price'] - data['Buying price']
        data['DayOfWeek'] = data['Date'].dt.day_name()

        # Calculate profit for each day of the week
        day_profit = data.groupby('DayOfWeek')['Profit'].sum()
        most_profitable_day = day_profit.idxmax()

        result = {"most_profitable_day": most_profitable_day}
        return JSONResponse(content=result)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
