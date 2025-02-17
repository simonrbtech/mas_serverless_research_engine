# Tutorial to build a Crewai based multi-agent system with Azure Functions hosting

This Python project demonstrates how to use the crewai library to set up a customized multi-agent research tool. Serverless Azure Functions will be used to deploy this engine and let it run in an automated fashion. The detailed tutorial and related blog posts can be found [here](https://simon.richebaecher.org/scalable-agents-tutorial). 

## Prerequisites

- Python 3.11 or higher
- IDE like Visual Studio Code
- Virtual environment (e.g., `venv` or `virtualenv`)

## Environment installation, testing and code deployment

1. Clone the repository to your local machine:

    ```bash
    git clone https://github.com/simonrbtech/mas_serverless_research_engine.git
    cd mas_serverless_research_engine
    ```

2. Create a virtual environment:

    ```bash
    virtualenv venv -p python3
    ```

3. Activate the virtual environment:

    - On macOS and Linux:

    ```bash
    source venv/bin/activate
    ```

    - On Windows:

    ```bash
    .\venv\Scripts\activate
    ```

4. Install project dependencies from `requirements.txt`:

    ```bash
    pip install -r requirements.txt
    ```

5. Set up an Azure Blob Storage and insert your storage connection string, your OpenAI API key and Serper API key to the local.settings.json as described (here)[http://simon.richebaecher.org/serverless-orchestration-context].

6. Test function app locally:

    ```bash
    func start
    ```

7. Create a Function App in Azure and deploy the code, for ex. by using Visual Studio Code plugins as described (this tutorial)[https://www.youtube.com/watch?v=lpZCwzYVNpA] by John Savill. Remember to stop or decommission the Function App when it is not in use anymore.  


## License

This project is licensed under the MIT License. For details, please refer to the [LICENSE](LICENSE) file.

The MIT License is a permissive open-source license that allows you to use, modify, and distribute this code for both personal and commercial purposes. It's a common choice for open-source projects due to its simplicity and flexibility.

For the full text of the MIT License, you can visit the official [MIT License page](https://opensource.org/licenses/MIT).
