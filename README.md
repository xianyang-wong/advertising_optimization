# advertising_optimization

### Problem
* The Company Milton advertises their new products through placing ad banners on five websites
* The company has the task of selecting (on a daily basis) the most effective advertising plan to reach the largest audience, i.e. maximize the number of user clicks, at the same time meeting the daily budget.
* They have 6 ad banners (same size but with different designs) and 5 websites to choose from. The advertising plan needs to determine which type of ad banner is displayed on which website and for what duration.
* If selected, each website can only display one banner and each banner can only be assigned to one website on the same day.

### Additional Information
* Factors influencing the user clicks
    * The popularity of each website is different, hence the number of user clicks acheived may be different for each banner.
    * The number of user clicks also depends on type of the ad banner, i.e. the design of banner.
    * The number of user clicks also depends to a lesser degree on the time of day that the banner is displayed. For example, the evening period from 7pm to 10pm attracts more user clicks.
* Unfortunately, there is no known formula for computing the user clicks on the ad banners. In practice, the actual user clicks achieved is only known at the end of the day.
* Records have been kept over the last three years showing the actual user clicks associated with each daily advertising plan (assignment of banners to websites, display durations).
* Total advertising cost can be computed as TotalCost = sum(duration in hours * cost per hour)
* The cost table is shown below (measured in dollars per hour).

| | Website 1 | Website 2 | Website 3 | Website 4 | Website 5 |
| :------------: |:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| Cost ($/hour)    | 15 | 10 | 8 | 8 | 12 |

* An example advertising plan is shown below

| Website | Banner | Start Time | Duration (hrs) | Cost |
|:-----------:|:-----------:|:-----------:|:-----------:|:-----------:|
| W1 | B5 | 5 | 9 | 135 |
| W2 | B6 | 14 | 5 | 50 |
| W3 | B4 | 10 | 3.5 | 28 |
| W4 | - | - | - | - |
| W5 | B1 | 13 | 5 | 60 |

### Objective
* Develop a hybrid intelligent system that can find an assignment of banners to websites and start time and duration for each banner's display time which will maximize the user clicks while ensuring that the budget requirement is met.
* Constraints
    * Assume the company has a budget of $300 per day.
    * Each banner can be displayed any time from 0:00 to 24:00 hours but we only consider a daily plan which means start time + duration must be <= 24:00.
    * Each banner can be displayed only one continuous period a day.
    * Duration = 0 means the banner is not displayed on the website

### Instructions to Generate Optimal Marketing Plan

Step1: Place training data features in .csv(s) format as 'data/WS1Data.csv'.\
Step2: Install necessary libraries.\
`pip install -r requirements.txt`\
Step3: Run program to get the outputs generated to the output folder.\
`python advertising_optimize.py --number_of_generations 100`
