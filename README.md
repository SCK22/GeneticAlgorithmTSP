# GeneticAlgorithmTSP
## Genetic algorithm code for solving Travelling Salesman Problem

Programming Language : Python

Number of cities : 11

General flow of solving a problem using Genetic Algorithm

                {
                  initialize population;
                  evaluate population;
                  while TerminationCriteriaNotSatisfied
                      {
                        select parents for reproduction;
                        perform recombination and mutation;
                        evaluate population;
                      }
                }

## run the following lines in terminal before proceeding :
    pip install -r requirements.txt
    conda install basemap
  


# pass this data frame to the genetic algorithm function as dist_mat
    data = pd.read_csv("data/cities_and_distances.csv")
    data.reset_index(inplace=True)
    data1 = data.iloc[:,2:]
    data1.index = data1.columns.values
    data1
Initialize population:
The initial population is a set of random routes generated using numpy.

<img src = "img/route_generation.PNG">

Evaluate population:
The evaluation is a process of finding how good the solutions is. This is <img src = "img/initial_population_cost.PNG">

Mutation:

<img src = "img/mutation.gif">

<img src = "img/mutation.PNG" >

Crossover:
Implemented PMX by goldberg - https://www.hindawi.com/journals/cin/2017/7430125/

<img src = "img/pmxcrossover_exp.jpg" >

OverallRun:

<img src = "img/overall_run.PNG" >

## The solution obtained from running Genetic algorithm

Starting:

<img src = "img/start.png" >

Final:

<img src = "img/final.PNG" >

_Note:_

_1. The final solution was obtained after multiple runs of the Genetic Algorithm with different inital population sizes and overall runs._

_2. The map needs access to city_lat_lon data, in the code to the file that is availabe in the data folder, if you are running the code from a different folder, the path should be changed accordingly._
