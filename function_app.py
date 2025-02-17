import os
import logging
import azure.functions as func

from crewai import Process, Crew

from crew import (projection_researcher, projection_analyst, roi_calculator, roi_reporter,
task_research_cost, task_research_revenue, task_analyze_deposit, task_roi_calculation, task_roi_plot_deposit)


app = func.FunctionApp()

@app.function_name(name="gatherAnalyzeCosts")
@app.timer_trigger(schedule="0 * * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def gatherAnalyzeCosts(myTimer: func.TimerRequest) -> None:

    crew_cost = Crew(
        agents=[projection_researcher, projection_analyst],
        tasks=[task_research_cost, task_analyze_deposit],
        verbose=True,
        process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )
    crew_cost.kickoff()

    if myTimer.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function executed.')


@app.function_name(name="gatherAnalyzeRevenue")
@app.timer_trigger(schedule="0 * * * * *", arg_name="myTimer2", run_on_startup=True,
              use_monitor=False) 
def gatherAnalyzeRevenue(myTimer2: func.TimerRequest) -> None:

    crew_revenue = Crew(
        agents=[projection_researcher, projection_analyst],
        tasks=[task_research_revenue, task_analyze_deposit],
        verbose=True,
        process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )
    crew_revenue.kickoff()

    if myTimer2.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function executed.')


@app.function_name(name="roiReport")
@app.timer_trigger(schedule="0 * * * * *", arg_name="myTimer3", run_on_startup=False,
              use_monitor=False) 
def roiReport(myTimer3: func.TimerRequest) -> None:

    crew_roi = Crew(
        agents=[roi_calculator, roi_reporter],
        tasks=[task_roi_calculation, task_roi_plot_deposit],
        verbose=True,
        process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
    )
    crew_roi.kickoff()

    if myTimer3.past_due:
        logging.info('The timer is past due!')

    logging.info('Python timer trigger function executed.')


# Blob trigger based alternative for "roiReport"
# @app.function_name(name="roiReport")
# @app.blob_trigger(arg_name="myblob", path="projections/{name}.json",
#                                connection="BlobStorageConnection") 
# def roiReport(myblob: func.InputStream):
#     logging.info(f"A new blob arrived and triggered the ROI Processing. "
#             f"Filename: {myblob.name} "
#             f"Blob Size: {myblob.length} bytes")

#     crew_roi = Crew(
#         agents=[roi_calculator, roi_reporter],
#         tasks=[task_roi_calculation, task_roi_plot_deposit],
#         verbose=True,
#         process=Process.sequential,  # Sequential process will have tasks executed one after the other and the outcome of the previous one is passed as extra content into this next.
#     )
#     crew_roi.kickoff()
    
#     logging.info(f"Python blob trigger function executed.")