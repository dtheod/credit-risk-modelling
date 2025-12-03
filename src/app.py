from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional
from contextlib import asynccontextmanager
import joblib
import pandas as pd
import numpy as np
import sys
import os

# Dirty way for this to work
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import transformers  # noqa: F401

# Global variable to store the pipeline
pipeline = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global pipeline
    try:
        pipeline = joblib.load("./models/inference_pipeline.joblib")
    except Exception as e:
        print(f"Error loading pipeline: with exception{e}")
        raise
    
    yield
    
    # Shutdown: Clean up resources if needed
    print("Shutting down...")

app = FastAPI(
    title="Credit Risk ML API for PD Model",
    description="API for predicting probability of default on loans",
    version="0.1.0",
    lifespan=lifespan
)


class LoanRequest(BaseModel):
    """Schema for loan application data"""
    LoanId: Optional[str] = Field(None, description="Loan ID")
    NewCreditCustomer: Optional[str] = Field(None, description="Whether customer is new to credit")
    VerificationType: Optional[int] = Field(None, description="Type of verification")
    LoanDuration: Optional[int] = Field(None, description="Loan duration in months")
    IncomeFromPrincipalEmployer: Optional[float] = Field(None, description="Income from principal employer")
    IncomeFromPension: Optional[float] = Field(None, description="Income from pension")
    IncomeFromFamilyAllowance: Optional[float] = Field(None, description="Income from family allowance")
    IncomeFromSocialWelfare: Optional[float] = Field(None, description="Income from social welfare")
    IncomeFromLeavePay: Optional[float] = Field(None, description="Income from leave pay")
    IncomeFromChildSupport: Optional[float] = Field(None, description="Income from child support")
    IncomeOther: Optional[float] = Field(None, description="Income from other sources")
    EmploymentDurationCurrentEmployer: Optional[str] = Field(None, description="Duration at current employer")
    IncomeTotal: Optional[float] = Field(None, description="Total income")
    Education: Optional[float] = Field(None, description="Education level")
    EmploymentStatus: Optional[float] = Field(None, description="Employment status")
    ExistingLiabilities: Optional[int] = Field(None, description="Number of existing liabilities")
    LiabilitiesTotal: Optional[float] = Field(None, description="Total liabilities")
    LanguageCode: Optional[int] = Field(None, description="Language code")
    Amount: Optional[float] = Field(None, description="Loan amount applied for")
    Age: Optional[int] = Field(None, description="Age of applicant")
    Interest: Optional[float] = Field(None, description="Interest rate")
    UseOfLoan: Optional[int] = Field(None, description="Purpose of loan")
    CreditScoreEsMicroL: Optional[str] = Field(None, description="Credit score category")
    AmountOfPreviousLoansBeforeLoan: Optional[float] = Field(None, description="Number of previous loans")
    PreviousEarlyRepaymentsCountBeforeLoan: Optional[float] = Field(None, description="Number of previous early repayments")
    PreviousRepaymentsBeforeLoan: Optional[float] = Field(None, description="Number of previous repayments")
    Country: Optional[str] = Field(None, description="Country code")
    Gender: Optional[float] = Field(None, description="Gender (0=Male, 1=Female)")
    MonthlyPayment: Optional[float] = Field(None, description="Monthly payment amount")
    HomeOwnershipType: Optional[int] = Field(None, description="Home ownership type")
    NoOfPreviousLoansBeforeLoan: Optional[float] = Field(None, description="Number of previous loans")


class PredictionResponse(BaseModel):
    """Schema for prediction response"""
    default_probability: float = Field(..., description="Probability of default (0-1)")
    prediction: int = Field(..., description="Binary prediction (0=No Default, 1=Default)")
    risk_level: str = Field(..., description="Risk level category")


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "message": "Credit Risk ML API is running",
        "status": "healthy",
        "model": "XGBoost PD Model"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_default(loan: LoanRequest):
    """
    Predict probability of default for a loan application
    
    Args:
        loan: Loan application data
        
    Returns:
        Prediction response with probability and risk level
    """
    global pipeline
    
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Convert request to DataFrame
        loan_data = pd.DataFrame([loan.model_dump()])
        
        # Make prediction using the pipeline
        prediction = pipeline.transform(loan_data)[0]
        
        # Determine risk level based on probability
        if prediction < 0.3:
            risk_level = "Low Risk"
        elif prediction < 0.6:
            risk_level = "Medium Risk"
        else:
            risk_level = "High Risk"
        
        return PredictionResponse(
            default_probability=float(prediction),
            prediction=int(prediction >= 0.5),
            risk_level=risk_level
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")