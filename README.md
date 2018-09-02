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

Initialize population:
The initial population is a set of random routes generated using numpy.

Evaluate population:
The evaluation is a process of finding how good the solutions is. This is <img src = "img/0220-2017-04-03.jpg">

