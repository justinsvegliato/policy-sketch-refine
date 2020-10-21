# policy-sketch-refine
Execute the following command:
`python3 earth-observation-example.py`

----------

## Examples

1. Set up a data directory on a disk partition large enough to store large datasets.

2. Set up your CSV configurations files. Look at the example ones.

3. Run all the experiments in a config file as shown below.

First, create all the abstractions from the config file:
```
python3 src/run.py src/experiments/earth_observation/config.csv <path-to-data-dir> abstract
```

Then, run all the simulations from the same config file:
```
python3 src/run.py src/experiments/earth_observation/config.csv <path-to-data-dir> simulate
```

If you want to force simulating everything *again* then add -f=1 to the command line:
```
python3 src/run.py src/experiments/earth_observation/config.csv <path-to-data-dir> simulate -f=1
```

----------

## Plot

Look at the examples in plot_examples.py

Run the examples with:
```
python src/plot_examples.py src/experiments/earth_observation/configi_ground.csv src/plot_examples.py src/experiments/earth_observation/config.csv <path-to-data-dir>
```

----------

## Best Practice

* Create different CSV config files, one for each "experiment batch". For example, change the grid size in one,
and the n of POIs in a different one, etc.

* Remember to duplicate rows with different domain variations (numbers 1-10) and simulation variations
(numbers 1-10). Two runs with the same domain and simulation variations will be deterministically the same.
There will only be a natural small variance in the runtime.


MASTER COMMAND TO RUN EVERYTHING:

python3 run.py experiments/earth_observation/config_0.csv results abstract && python3 run.py experiments/earth_observation/config_0.csv results simulate && python3 run.py experiments/earth_observation/config_1.csv results simulate && python3 run.py experiments/earth_observation/config_2.csv results simulate && python3 run.py experiments/earth_observation/config_ground.csv results simulate
