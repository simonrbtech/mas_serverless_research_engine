import os
from dotenv import load_dotenv
load_dotenv()

from crewai import Agent, Task, Process, Crew
from crewai_tools import FileWriterTool, SerperDevTool

######################### Initialization ###################

api = os.getenv("OPENAI_API_KEY")

######################### Tools  ###########################

web_search_tool = SerperDevTool(
    n_results=20
)
file_writer_tool = FileWriterTool(directory='./output')

######################### Agents ###########################

researcher = Agent(
    role="Researcher",
    goal="Find all monetary information for a given product category and year",
    backstory="""
    You are an expert in identifying relevant monetary factors and their numerical values for a target timeframe.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[web_search_tool]
)

analyst = Agent(
    role="Analyst",
    goal="""Generate an month-by-month timeseries from the provided historical data.""",
    backstory="""
    You specialize in timeseries transformation and mathematical models to fill gaps in historical data.
    """,
    verbose=True,
    allow_delegation=True,
    tools=[file_writer_tool]
)

######################### Tasks ###########################

task_research = Task(
    description="""
    Search for costs associated with owning an e-bike in the year 2024 ONLY
    using the query: 'e-bike ownership cost breakdown 2024'. 
    Extract all data that can be used to build any kind of timeseries, including
    timestamps or seasonal indicators. You only hand over a result that contains numerical
    cost values with associated timestamps or seasonal indicators. 
    """,
    expected_output="Collection cost information for the year 2024.",
    agent=researcher,
    human_input=False,
)

task_analyze_deposit = Task(
    description="""
    Analyze the information given by the researcher and create timeseries for each cost factor with one value per month. 
    If you do not have enough datapoints you can use the ARIMA method to create estimates with the datapoints you have. 
    In the end you should have one value per month for each cost factor.
    From such create single JSON file that contains all timeseries. 
    """,
    expected_output="A JSON file with timeseries written to the folder named output.",
    agent=analyst,
    human_input=False,
)

######################### Execution ###########################

my_crew = Crew(
    agents=[researcher, analyst],
    tasks=[task_research, task_analyze_deposit],
    verbose=True,
    process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
)

result = my_crew.kickoff()

print("######################")
print(result)