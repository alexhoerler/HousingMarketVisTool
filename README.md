# Housing Market Visualization Tool
Within the dynamic landscape of housing investments, where strategic decisions can dictate financial prosperity or adversity, our mission is to deliver a cutting-edge predictive-model-driven visualization tool that sheds light on location-specific housing market trends. While platforms such as Zillow and Redfin offer property details such as estimated value and rent potential, they lack the ability to extract location effects on house prices and trends, making aggregate analysis very difficult. Our project aims to bridge this gap by developing a geographic heatmap integrated with predictive models on aggregate trends for better risk assessment and informed decision-making. From savvy homeowners and investors to governments, the potential beneficiaries are myriad. By equipping users with market trends and foresight, we facilitate not only informed residential choices and lucrative investments but also enable governments to optimize resource allocation and plan for societal housing needs effectively. Furthermore, through data-driven insights, we empower users to navigate the complex housing landscape with confidence, fostering smarter decisions and deeper market comprehension.

# Installation / Set Up
## Online Use (Github Pages)
1. Go to https://alexhoerler.github.io/HousingMarketVisTool/
2. Explore HousingMarketVisTool to your heart's content!

## Local Installation
1. Clone the HousingMarketVisTool Repository
```
$ git clone https://github.com/alexhoerler/HousingMarketVisTool.git
```
2. Open the Project in VSCode
3. Ensure that Python 3.x is installed on your system
```
$ python3 --version
```
4. Run the HTTP Server in the project directory
```
$ python3 -m http.server 8000
```
5. Click on the generated link to connect to the server running on port 8000
6. Explore HousingMarketVisTool to your heart's content!

# Execution
The website will appear as follows upon opening:
![Example Image](images/initial_page.png)
## Interactive Feature 1
Users can click on different states to visualize housing market data.
![Example Image](images/GA.png)
![Example Image](images/MI.png)
## Interactive Feature 2
The line graph employs predictive ML (machine learning) algorithms to forecast housing prices. Users can hover over the data points to view detailed price information.
![Example Image](images/line-graph.png)