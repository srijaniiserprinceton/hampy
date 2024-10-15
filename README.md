In order to get started, you need to have a Python environment with all the packages listed in `pyspeads_env.yml`.  The required directory structure can be build by running the following lines in your command line
```
mkdir data plots
```
To get the interactive `hammerfinder` plot execute the following line in your command line
```
python interactive_hammerchecks.py
```
It should load an interactive plotting widget where you can change the time by clicking the time index bar at the bottom of the plot.
In order to change the default time it opens, you can change the following lines in `interactive_hammerchecks.py`
```
if __name__=='__main__':
    # user defined date and time
    year, month, date = 2020, 1, 29
    hour, minute, second = 18, 10, 1  
```





