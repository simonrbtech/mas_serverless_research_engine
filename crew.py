import io
import os
import json
import logging
from collections import defaultdict  
from typing import List, Dict, Optional
from pydantic import BaseModel

import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
matplotlib.use('Agg')  # Use non-GUI backend for rendering plots
from datetime import datetime

from azure.storage.blob import BlobServiceClient

from langchain.tools import tool
from crewai import Agent, Task 
from crewai_tools import SerperDevTool


######################### Initialization ###########################


api = os.environ.get("OPENAI_API_KEY")

blob_connection_string = os.environ.get("BlobStorageConnection")
container_name_projection = "projections"
container_name_report = "reports"

web_search_tool = SerperDevTool(
    # search_url="https://google.serper.dev/scholar",
    country="de",
    locale="de",
    n_results=10,
)


######################### Tools & Schemas ###########################


# Configure logging to print to the console
logging.basicConfig(
    level=logging.ERROR,  # Log only errors and higher levels
    format="%(asctime)s - %(levelname)s - %(message)s",  # Log message format
)

@tool("Error Logging")
def error_logging_tool(error_message: str):
    """
    Logs an error message and prints it to the console.
    
    :param error_message: The error message to log and display.
    """
    try:
        # Log the error
        logging.error(error_message)
        
        # Print the error to the console
        print(f"Logged Error: {error_message}")
        
        return f"Error logged: {error_message}"
    except Exception as e:
        # Log any unexpected error during the logging process
        logging.error(f"Unexpected error in logging tool: {e}")
        return f"Failed to log error: {e}"


@tool("Blob Storage Directory Read")
def blob_directory_read_tool():
    """
    Lists blobs in a specific container of Azure Blob Storage.

    :param input: BlobDirectoryReadInput containing container name and optional prefix.
    """

    try:
        # Initialize BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(container_name_projection)

        # List blobs in the container with optional prefix
        blob_list = container_client.list_blobs()
        blobs = [blob.name for blob in blob_list]

        return blobs  # Returns a list of blob names
    except Exception as e:
        return f"Failed to list blobs: {str(e)}"



@tool("Blob Storage File Read")
def blob_file_read_tool(file_name: str):
    """
    Reads the content of a specific file from an Azure Blob Storage. 

    :param file_name: single file name as a string.
    """

    try:
        # Initialize BlobServiceClient
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(container_name_projection)

        # Get the blob client
        blob_client = container_client.get_blob_client(file_name)

        # Download blob content
        download_stream = blob_client.download_blob()
        content = download_stream.readall().decode("utf-8")  # Decodes content as a string

        return content  # Returns the content of the blob
    except Exception as e:
        return f"Failed to read blob: {str(e)}"


class UploadInput(BaseModel):
    json_data: dict  # The JSON data to upload
    file_name: str   # The name of the blob (file)


@tool("Json Blob storage upload")
def upload_json_to_azure_blob(input: UploadInput):
    """
    Uploads JSON data to Azure Blob Storage.
    
    :param input: UploadInput object containing json_data and file_name.
    """

    try:
        # Create the BlobServiceClient object
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        
        # Get the container client
        container_client = blob_service_client.get_container_client(container_name_projection)
        
        # Ensure the container exists
        if not container_client.exists():
            container_client.create_container()

        # Convert the JSON data to a string
        json_string = json.dumps(input.json_data, indent=4)
        
        # Upload the JSON string as a blob
        blob_client = container_client.get_blob_client(input.file_name)
        blob_client.upload_blob(json_string, overwrite=True)
        
        return f"JSON data uploaded successfully to blob: {input.file_name} in container: {container_name_projection}"

    except Exception as e:
        return f"Failed to upload JSON to Azure Blob Storage: {str(e)}"


class MultiSeriesPlotInput(BaseModel):
    data: dict  # Dictionary of {series_name: [data_points]}
    timestamps: list  # Timestamp strings corresponding to the data points
    roi: float # Single ROI value 
    filename: str = "multi_series_plot.png"  # Name of the output file (optional)


@tool("Timeseries plotting and saving to Blob Storage")
def plot_and_save(input: MultiSeriesPlotInput):
    """
    Plots multiple time series and uploads the plot as a PNG file to Azure Blob Storage.
    
    :param input: MultiSeriesPlotInput object containing data, timestamps, roi, and filename.
    """
    try:
        # Apply a dark background theme
        plt.style.use("dark_background")

        # Print overall timestamps length (for debugging)
        timestamps_length = len(input.timestamps)
        print(f"Number of timestamps: {timestamps_length}")

        # Create the plot in memory, set dark figure background
        fig, ax = plt.subplots(figsize=(12, 8))

        # Check each series dimension vs. timestamps
        for series_name, series_data in input.data.items():
            series_length = len(series_data)
            print(f"Series '{series_name}': has {series_length} data points.")

            # Check if there's a mismatch
            if series_length != timestamps_length:
                print(
                    f"WARNING: Dimension mismatch for series '{series_name}' - "
                    f"expected {timestamps_length} but got {series_length}."
                )
                # Optionally skip, pad, or raise an error. 
                # e.g.: continue

            ax.plot(input.timestamps, series_data, marker='o', label=series_name)

        # Rotate the x-axis tick labels vertically
        plt.xticks(rotation=90)

        # Set y-axis major ticks every 100k
        ax.yaxis.set_major_locator(ticker.MultipleLocator(100000))

        # Add plot details
        ax.set_title(f"Investment projection with a final ROI of {input.roi}")
        ax.set_xlabel("Timestamp")
        ax.set_ylabel("Value (increments of 100k)")
        ax.legend()  
        ax.grid(True, color='gray', linestyle=':', linewidth=0.5)

        # Save the plot to an in-memory file (BytesIO)
        img_stream = io.BytesIO()
        plt.savefig(img_stream, format="png")
        plt.close(fig)
        img_stream.seek(0)  # Reset the stream position

        # Upload to Azure Blob Storage
        blob_service_client = BlobServiceClient.from_connection_string(blob_connection_string)
        container_client = blob_service_client.get_container_client(container_name_report)

        # Ensure the container exists
        if not container_client.exists():
            container_client.create_container()

        # Upload the image as a blob
        blob_client = container_client.get_blob_client(input.filename)
        blob_client.upload_blob(img_stream, blob_type="BlockBlob", overwrite=True)

        return f"Plot uploaded successfully to container '{container_name_report}' as blob '{input.filename}'"

    except Exception as e:
        return f"Failed to upload plot to Azure Blob Storage: {str(e)}"


class ROIInputSchema(BaseModel):

    Purchase_costs: List[float]
    Purchase_dates: List[str]
    Operational_and_maintenance_costs: List[float] 
    Operational_and_maintenance_dates: List[str]
    Service_revenue: List[float]
    Service_revenue_dates: List[str]

    class Config:
        json_schema_extra = {
            "example": {
                "Purchase_costs": [1000.0, 950.0, 900.0, 850.0],
                "Purchase_dates": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01"
                ],
                "Operational_and_maintenance_costs": [100.0, 95.0, 90.0, 85.0],
                "Operational_and_maintenance_dates": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01"
                ],
                "Service_revenue": [500.0, 510.0, 520.0, 530.0],
                "Service_revenue_dates": [
                    "2023-01-01",
                    "2023-02-01",
                    "2023-03-01",
                    "2023-04-01"
                ]
            }
        }


class ROIcalculator:
    @tool("ROIcalculator", args_schema=ROIInputSchema)
    def calculate_roi(
        Purchase_costs: List[float],
        Purchase_dates: List[str],
        Operational_and_maintenance_costs: List[float],
        Operational_and_maintenance_dates: List[str],
        Service_revenue: List[float],
        Service_revenue_dates: List[str],
    ):
        """
        Calculate the Return on Investment (ROI) using provided cost and revenue time series data.

        Parameters:
            Purchase_costs (List[float]): List of purchase costs.
            Purchase_dates (List[str]): Corresponding dates for purchase costs.
            Operational_and_maintenance_costs (List[float]): List of summed up operational and maintenance costs.
            Operational_and_maintenance_dates (List[str]): Corresponding dates in %Y-%m-%d format for those costs.
            Service_revenue (List[float]): List of revenue generated from service.
            Service_revenue_dates (List[str]): Corresponding dates for service revenue.

        Returns:
            dict: A dictionary *directly matching* the MultiSeriesPlotInput schema:
                  {
                    "data": {
                      "Purchase_costs": [...],
                      "Op_main_costs": [...],
                      "Total_costs": [...],
                      "Revenue": [...]
                    },
                    "timestamps": [...],  # monthly labels like "YYYY-MM"
                    "roi": <float>,
                    "filename": "multi_series_plot.png"
                  }
        """

        def to_month(date_input):
            """Convert a date string (YYYY-MM-DD) or datetime object to 'YYYY-MM'."""
            if isinstance(date_input, str):
                dt = datetime.strptime(date_input, "%Y-%m-%d")
            elif isinstance(date_input, datetime):
                dt = date_input
            else:
                raise ValueError(f"Unsupported type for to_month: {type(date_input)}")
            return dt.strftime("%Y-%m")

        # Organize data into monthly buckets
        purchase_costs_by_month = defaultdict(list)
        op_main_costs_by_month = defaultdict(list)
        revenue_by_month = defaultdict(list)

        for date, cost in zip(Purchase_dates, Purchase_costs):
            purchase_costs_by_month[to_month(date)].append(cost)

        for date, cost in zip(Operational_and_maintenance_dates, Operational_and_maintenance_costs):
            op_main_costs_by_month[to_month(date)].append(cost)

        for date, rev in zip(Service_revenue_dates, Service_revenue):
            revenue_by_month[to_month(date)].append(rev)

        # Prepare structure to hold monthly results
        monthly_data = {
            "Month": [],
            "Purchase_costs": [],
            "Op_main_costs": [],
            "Total_costs": [],
            "Revenue": [],
        }

        # Gather all unique months in chronological order
        all_months = sorted(
            set(purchase_costs_by_month.keys())
            | set(op_main_costs_by_month.keys())
            | set(revenue_by_month.keys())
        )

        # Fill in monthly breakdown
        for month in all_months:
            purchase_cost = sum(purchase_costs_by_month[month])
            op_main_cost  = sum(op_main_costs_by_month[month])
            total_cost    = purchase_cost + op_main_cost
            revenue       = sum(revenue_by_month[month])

            monthly_data["Month"].append(month)
            monthly_data["Purchase_costs"].append(purchase_cost)
            monthly_data["Op_main_costs"].append(op_main_cost)
            monthly_data["Total_costs"].append(total_cost)
            monthly_data["Revenue"].append(revenue)

        # Compute the overall ROI
        overall_total_cost = sum(monthly_data["Total_costs"])
        overall_revenue    = sum(monthly_data["Revenue"])
        if overall_total_cost != 0:
            overall_roi = (overall_revenue - overall_total_cost) / overall_total_cost
        else:
            overall_roi = 0.0

        return {
            "data": {
                "Purchase_costs": monthly_data["Purchase_costs"],
                "Op_main_costs": monthly_data["Op_main_costs"],
                "Total_costs": monthly_data["Total_costs"],
                "Revenue": monthly_data["Revenue"]
            },
            "timestamps": monthly_data["Month"],
            "roi": overall_roi,
            "filename": "roi_report.png"
        }


######################### Agents ###########################


projection_researcher = Agent(
    role="Researcher",
    goal="Find variables in a given task that are require future projections and search for sensible data on the web.",
    backstory="""
    You are an expert in identifying missing data for future projections in a task and searching for suitable 
    substitutes on the web.  
    """,
    # Core parameters to reduce randomness
    # openai_model="gpt-4-turbo",
    openai_temperature=0,                    # zero temperature
    openai_top_p=1,                          # full sampling range (equivalent to no nucleus sampling)
    openai_frequency_penalty=0,              # no penalty for repeated tokens
    openai_presence_penalty=0,               # no presence penalty
    openai_best_of=1,                        # only one generation

    verbose=True,
    allow_delegation=True,
    tools=[web_search_tool, error_logging_tool]
)

projection_analyst = Agent(
    role="Analyst",
    goal="""Analyze given projection data for cost and revenue data and summarize or transform it so that it fits into the JSON format defined by the ROIInputSchema.""",
    backstory="""
    You are responsible for creating properly structured input data for ROI calculations.
    If given raw data, you process it and package it into the expected JSON format.
    """,

    # Core parameters to reduce randomness
    openai_temperature=0,                    # zero temperature
    openai_top_p=1,                          # full sampling range (equivalent to no nucleus sampling)
    openai_frequency_penalty=0,              # no penalty for repeated tokens
    openai_presence_penalty=0,               # no presence penalty
    openai_best_of=1,                        # only one generation

    verbose=True,
    allow_delegation=True,
    tools=[upload_json_to_azure_blob, error_logging_tool]
)

roi_calculator = Agent(
    role="ROI analyst and calculator",
    goal="""Extract relevant ROI timeseries data for a certain product and substitute missing data as described in the task. 
    With a complete dataset that ranges over the years 2025 until 2030 you can calculate the ROI output.""",
    backstory="""
    You are an expert in identifying missing data for the ROI calculations and replacing them with suitable substitutes.
    """,
    # Core parameters to reduce randomness
    openai_temperature=0,                    # zero temperature
    openai_top_p=1,                          # full sampling range (equivalent to no nucleus sampling)
    openai_frequency_penalty=0,              # no penalty for repeated tokens
    openai_presence_penalty=0,               # no presence penalty
    openai_best_of=1,                        # only one generation 

    verbose=True,
    allow_delegation=True,
    tools=[blob_directory_read_tool, blob_file_read_tool, ROIcalculator.calculate_roi, error_logging_tool]
)

roi_reporter = Agent(
    role="ROI Reporter",
    goal="""Take the dataframe or dictionary type timeseries data from ROI calculations, 
    and properly format it as an input for the plot_and_save tool to finalize plotting and saving it.""",
    backstory="""
    """,
    openai_temperature=0,                    # zero temperature
    openai_top_p=1,                          # full sampling range (equivalent to no nucleus sampling)
    openai_frequency_penalty=0,              # no penalty for repeated tokens
    openai_presence_penalty=0,               # no presence penalty
    openai_best_of=1,                        # only one generation   

    verbose=True,
    allow_delegation=True,
    tools=[plot_and_save, error_logging_tool]
)


######################### Tasks ###########################


task_research_cost = Task(
    description="""
    We are running an e-bike rental business and look for the costs associated with expanding it. 
    1) Search for the average purchase price, operation and maintenance costs ONLY with the search term 'e-bike ownership cost breakdown 2024'.
    2) Use those values to create TWO timeseries that range from the year 2025 to the end of 2030 with aggregate values FOR EACH MONTH - one for purchase and the other
    for combined operation and maintenance cost. In your estimates use the information that we plan to buy 700 e-bikes in January 2025. In each January from 2026 
    until 2030 we buy 100 new bikes. Monthly operational and maintenance costs per e-bike SHOULD NOT exceed 0.5 percent of the average purchase price. 
    
    Check: Your output should ALWAYS contain one value for each month from 2025 until 2030 and NO "..." assumptions. 
    """,
    expected_output="TWO monthly timeseries that range from the year 2025 to the end of 2030 on different cost factor projections for e-bikes.",
    agent=projection_researcher,
    human_input=False,
)

task_research_revenue = Task(
   description="""
    We are running an e-bike rental business and look for the revenue that could be earned by expanding it. 
    1) Estimate an average monthly rental  from data ONLY gathered with the search term 'e-bike rental monthly fee'.
    2) Use this average and the following instructions to create a revenue timeseries that ranges from the year 2025 to the end of 2030 with aggregate values FOR EACH MONTH:  
        - In your estimates use the information that we plan to have 600 loyal customers in the first year and that this number grows by 100 customers per year. 
        - Add 150.000€ on top of the december revenue of each year, which we can achieve by selling 100 of our older bikes at the end of each year on 
        the second-hand market. 
        - Finally at the end of 2030 we will sell the rest of the e-bike inventory for 500.000€.  
    
    Check: Your output should ALWAYS contain one value for each month from 2025 until 2030 and NO "..." assumptions. Add the keyword "Revenue" to your output.  
    """,
    expected_output="Monthly timeseries timeseries that ranges from the year 2025 to the end of 2030 on revenue projections for e-bike rental services.",
    agent=projection_researcher,
    human_input=False,
)

task_analyze_deposit = Task(
    description="""
    Analyze the projections / time series given by the researcher.
    From such create a python dictionary input with values that follow the ROIInputSchema. 
    If data for any key of the ROIInputSchema is missing you leave an empty list [].

    ROIInputSchema:
        Purchase_costs (List[float]): List of purchase costs.
        Purchase_dates (List[str]): Corresponding dates for purchase costs.
        Operational_and_maintenance_costs (List[float]): List of summed up operational and maintenance costs. 
        Operational_and_maintenance_dates (List[str]): Corresponding dates in %Y-%m-%d format for operational and maintenance costs.
        Service_revenue (List[float]): List of revenue generated from service.
        Service_revenue_dates (List[str]): Corresponding dates for service revenue.

    Finally upload the resulting dictionary to an azure blob storage (by using the upload_json_to_azure_blob tool)
    with either roi_cost_projection.json or roi_revenue_projection.json as suitable file_names by following these IMPORTANT instructions: 
    - When calling `upload_json_to_azure_blob`, pass this structure EXACTLY like this:
        {
            "input":{
                "json_data": {...},
                "file_name": "..."
            }
        }
    - The "file_name" string is ONLY delivered within the "input" brackets 
    - If you fail, retry up to 5 times, then terminate.
    """,
    expected_output="Dictionary uploaded as a JSON to azure blob storage with either roi_cost_projection.json or roi_revenue_projection.json as suitable names.",
    agent=projection_analyst,
    human_input=False,
)

task_roi_calculation = Task(
    description="""
    Extract all timeseries data for either roi_cost_projection or roi_revenue_projection cases from the available blob directory by using 
    both the blob_directory_read_tool and blob_file_read_tool. Once you committed to one projection case do NOT read data related to the other case.
    If you find empty values and timeseries lists use the history.json to substitute them by doing an estimation with methods like ARIMA
    for all of the months from 2025 until 2030 with the available historical values. 
    
    Once you have a complete dataset following the ROIInputSchema you can start to calculate the ROI for the product/service calculate_roi tool.
    ROIInputSchema:
        Purchase_costs (List[float]): List of material costs.
        Purchase_dates (List[str]): Corresponding dates for material costs.
        Operational_and_maintenance_costs (List[float]): List of person days (HR related) costs.
        Operational_and_maintenance_dates (List[str]): Corresponding dates in %Y-%m-%d format for person days costs.
        Service_revenue (List[float]): List of revenues.
        Service_revenue_dates (List[str]): Corresponding dates for revenues.
    
    Hand over the result of calculate_roi AS-IS to the next agent. 
    """,
    expected_output="Dataframe with monthly timeseries values for the years 2025 up until the end of 2030 plus an ROI value.",
    agent=roi_calculator,
    human_input=False,
)

task_roi_plot_deposit = Task(
    description="""
    Check that the input from the ROI calculator conforms to the MultiSeriesPlotInput structure. 
    MultiSeriesPlotInput(BaseModel):
        data: dict  # Dictionary of {series_name: [data_points]}
        timestamps: list  # Common timestamps for all series
        roi: float # Single ROI value  
        filename: str = "multi_series_plot.png"  # Name of the output file (optional)
    
    If it does feed it into the plot_and_save tool to generate a plot and save it afterwards. Then terminate your task.
    If you are not able to complete the save process after 5 attempts terminate the task.
    """,
    expected_output="Generated and saved plot",
    agent=roi_reporter,
    human_input=False,
)