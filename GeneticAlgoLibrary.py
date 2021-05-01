# GeneticAlgorithmTSP#
import ray # for efficient multiprocessing
import warnings
import numpy as np
import pandas as pd
import random  # library to generate random numbers

np.random.seed(seed=42)
import matplotlib.pyplot as plt  # For plotting
from numpy import math as m
from tqdm import tqdm

class GeneticAlgorithmTSP:

    """This class implements all the methods required to apply genetic algorithm
     to Travelling Salesman Problem."""

    def __init__(
        self,
        number_of_cities,
        initial_pop_size,
        nelite,
        percentage_to_mutate,
        percentage_to_crossover,
        dist_mat,
    ):
        """Initialize the required values and import libraries"""

        self.number_of_cities = number_of_cities
        self.initial_pop_size = initial_pop_size
        self.nelite = nelite
        self.percentage_to_mutate = percentage_to_mutate
        self.percentage_to_crossover = percentage_to_crossover
        self.dist_mat = dist_mat
        self.cross_over_df = None
        self.elite_few_df = None

    # generate an initial random route
    def get_route(self) -> list:
        """ This function generates a route by sampling numbers."""
        np.random.seed(seed=42)
        my_randoms = random.sample(
            range(1, self.number_of_cities), self.number_of_cities - 1
        )  # excluding 0 from the range as 0 is the starting point for us i.e Bangalore
        return my_randoms

    """We define a fitness criteria (here the fitness criteria is the distance
     travelled if a particular route is taken)"""
    @ray.remote
    def fitness_function(self, route) -> int:
        """This functions takes the route generated and returns the cost incurred.
        Example :
                input : [8, 3, 5, 7, 1, 9, 10, 4, 6, 2]
                output : 17514

        Implementation:
                fitness_function([8, 3, 5, 7, 1, 9, 10, 4, 6, 2])
                output: 17514
        """
        traverse_data = self.dist_mat.copy()  # creating a copy of the original df
        source_point = 0  # defining a starting point
        stops_covered = 0  # setting the number of stops covered to 0
        route_cost = 0  # setting the initial cost to 0
        route1 = route  # + [pvRouteMap[0]]
        # initiate a while loop and calculate the cost for the whole path traversed and return the cost
        while stops_covered < len(route1):
            route_cost = route_cost + traverse_data.iloc[source_point, route1[stops_covered]]
            """route cost is the sum of the cost incurred from travelling from one point to the next according
			to the route that was generated previously."""
            source_point = route1[
                stops_covered
            ]  # update the source point to the next point in the route
            stops_covered = stops_covered + 1  # Adding 1 to the stops covered
        route_cost = route_cost + traverse_data.iloc[route1[-1], 0]
        return route_cost

    """Generate an initial population
     of random routes (generally we generate a large number of initial 
     routes)"""
    def initial_pop_cost(self) -> pd.DataFrame:
        """This function generates routes and calculates the cost of the routes.

        Returns: dataframe of routes and the cost."""
        np.random.seed(seed=42)
        ninitpop = self.initial_pop_size
        intial_cost = 0
        route_cost = []
        # generate ninitpop routes
        routes = [
            self.get_route() for i in range(0, ninitpop)
        ]  
        # calculate the fitness criteria
        routes_cost = ray.get([self.fitness_function.remote(self,i) for i in routes])
        routes_df = pd.DataFrame([routes, routes_cost]).transpose()
        routes_df.columns = ["Route", "Cost"]
        routes_df = routes_df.sort_values("Cost")
        self.sorted_population = routes_df
        return routes_df

    # Here we follow the elitist approach
    """Sort the routes based on the distance travelled (cost)
     and take the elite ones"""

    def the_elite_few(self) -> pd.DataFrame:
        """This functions picks 'nelite' number of best performing solutions.
        Takes input of population sorted based on the cost
        Returns: top 'nelite' solutions.
        """
        elite_few = self.sorted_population.head(self.nelite)
        self.elite_few_df = elite_few
        return elite_few

    def get_mutated_path(self, init_path: list, mutate_factor: int) -> list:
        """get_mutated_path takes an input path and a mutate factor,
         returns mutated path
        Args:
            init_path : initial path
            mutate_factor: index from where the solution parts should be swapped

        Retuns:
            returns mutated path
        """
        try:
            temp1 = init_path[0:mutate_factor]
            temp2 = init_path[mutate_factor : len(init_path)]
            newPath = temp2 + temp1
        except:
            temp1 = init_path[0 : max(mutate_factor)]
            temp2 = init_path[max(mutate_factor) :]
            newPath = temp2 + temp1
        return newPath

    def mutation_function(self, elite_few_df):
        """This function mutates n input routes where n is calculated based on 
        the percentage_to_mutate and returns the
        corresponding mutated solution and the cost."""
        # random number for mutate factor
        p = int(round(elite_few_df.shape[0] * self.percentage_to_mutate, 0))
        picked_route_maps = elite_few_df.Route.head(p).tolist()
        # pick a new index for every solution
        mutated_route_list = []
        for i in tqdm(picked_route_maps, desc= "Mutation"):
           mutated_route_list.append(self.get_mutated_path(i, random.choice(range(0, len(elite_few_df.Route[0]), 1))))
            
        mutated_routes_cost = ray.get([self.fitness_function.remote(self,i) for i in mutated_route_list])
        mutated_routes_df = pd.DataFrame(
            [mutated_route_list, mutated_routes_cost]
        ).transpose()
        mutated_routes_df.columns = ["Route", "Cost"]
        self.mutated_routes_df = mutated_routes_df
        return mutated_routes_df

    def cross_over_function(
        self, parent1, parent2, crossover_factor_start_pos=2, crossover_factor_end_pos=6
    ):
        """This function implements the partially mapped crossover by goldeberg
         (https://www.hindawi.com/journals/cin/2017/7430125/)
        inputs: 2 parent solutions, crossover start position, crossover end position.
        crossover start position and  crossover end position are randomly 
        generated in the solution.
        output: 2 child solutions, i.e the crossedover solutions.
        Example :
                Inputs:
                        p1 = [5,7,2,4,8,9,3,6]
                        p2 = [8,6,9,3,2,5,7,4]
                        crossover_start_pos 2 (randomly selected)
                        crossover_end_pos 6 (randomly selected)
                Internals:
                        The mapping created is [(2, 9), (4, 3), (8, 2), (9, 5)]
                        Child1 Intermediate [9, 7, 9, 3, 2, 5, 4, 6]
                        Child1 Intermediate [2, 7, 9, 3, 2, 5, 4, 6]
                        Child1 Intermediate [8, 7, 9, 3, 2, 5, 4, 6]
                        Child1 final [8, 7, 9, 3, 2, 5, 4, 6]
                        Child2 original [8, 6, 2, 4, 8, 9, 7, 4]
                        Child2 Intermediate [2, 6, 2, 4, 8, 9, 7, 3]
                        Child2 Intermediate [9, 6, 2, 4, 8, 9, 7, 3]
                        Child2 Intermediate [5, 6, 2, 4, 8, 9, 7, 3]
                        Child2 final [5, 6, 2, 4, 8, 9, 7, 3]
                Output:
                        Child: [8, 7, 9, 3, 2, 5, 4, 6]
                        Child: [5, 6, 2, 4, 8, 9, 7, 3].
        """
        indexes_for_crossover = random.sample(set(range(len(parent1))), 2)
        crossover_factor_start_pos, crossover_factor_end_pos = (
            min(indexes_for_crossover),
            max(indexes_for_crossover),
        )
        ## generate child 1
        child1 = (
            parent1[0:crossover_factor_start_pos]
            + parent2[crossover_factor_start_pos:crossover_factor_end_pos]
            + parent1[crossover_factor_end_pos:]
        )

        ## generate child 2
        child2 = (
            parent2[0:crossover_factor_start_pos]
            + parent1[crossover_factor_start_pos:crossover_factor_end_pos]
            + parent2[crossover_factor_end_pos:]
        )

        ## Create mappings
        mpping = list(
            zip(
                parent1[crossover_factor_start_pos:crossover_factor_end_pos],
                parent2[crossover_factor_start_pos:crossover_factor_end_pos],
            )
        )

        # run until all the nodes in the route are unique
        while len(np.unique(child1)) != len(child1):
            child1_part = (
                child1[:crossover_factor_start_pos] + child1[crossover_factor_end_pos:]
            )
            for i in child1_part:
                for j in mpping:
                    if i == j[1]:
                        child1_part[child1_part.index(i)] = j[0]

            child1 = (
                child1_part[:crossover_factor_start_pos]
                + child1[crossover_factor_start_pos:crossover_factor_end_pos]
                + child1_part[crossover_factor_start_pos:]
            )

        # run until all the nodes in the route are unique
        while len(np.unique(child2)) != len(child2):
            child2_part = (
                child2[:crossover_factor_start_pos] + child2[crossover_factor_end_pos:]
            )
            for i in child2_part:
                for j in mpping:
                    if i == j[0]:
                        child2_part[child2_part.index(i)] = j[1]
            child2 = (
                child2_part[:crossover_factor_start_pos]
                + child2[crossover_factor_start_pos:crossover_factor_end_pos]
                + child2_part[crossover_factor_start_pos:]
            )

        return (child1, child2)

    def routes_after_cross_over(self):
        """This functions takes in a population and performs crossover on the
         top_per records.
        output is a set of children after the crossover operation."""
        # taking the top_per% of this new df and using crossover
        sorted_pop = self.sorted_population
        tp = int(np.ceil(self.percentage_to_crossover / 100))
        if tp < 3:
            tp = 3
        top_few = sorted_pop.head(tp)
        routes_crossover = []
        ind = top_few.index.tolist()
        for i in tqdm(range(int(top_few.shape[0])), desc= "Crossover"):  # for every row randomly pick a pair to crossover
            try:
                indexes = random.sample(ind, 2)
                temp1, temp2 = (
                    top_few.iloc[top_few.index == indexes[0]].Route.tolist()[0],
                    top_few.iloc[top_few.index == indexes[1]].Route.tolist()[0],
                )
                sol1, sol2 = self.cross_over_function(temp1, temp2)
                routes_crossover.extend([sol1, sol2])
            except:
                pass
            # after doing crossover, remove the indeces from the data, continue with the remaining data
            try:
                ind.remove(indexes[0])
                ind.remove(indexes[1])
            except:
                pass
        cost_crossover = ray.get([self.fitness_function.remote(self,i) for i in routes_crossover])
        cross_over_df = pd.DataFrame(
            [routes_crossover, cost_crossover],
        ).transpose()
        cross_over_df.columns = ["Route", "Cost"]
        self.cross_over_df = cross_over_df
        return cross_over_df

class OverallGaRun(GeneticAlgorithmTSP):
    """Inherits GeneticAlgorithmTSP class"""
    # Start Ray.
    ray.init(ignore_reinit_error=True)
    def __init__(
        self,
        noverall,
        number_of_cities,
        initial_pop_size,
        nelite,
        percentage_to_mutate,
        percentage_to_crossover,
        dist_mat,
    ):
        super().__init__(
            number_of_cities,
            initial_pop_size,
            nelite,
            percentage_to_mutate,
            percentage_to_crossover,
            dist_mat,
        )
        self.noverall = noverall # overall runs of the GA
        cities_mapping = {}
        for i in enumerate(tqdm(self.dist_mat.columns,desc = "Cities Mapping")):
            cities_mapping[i[0]] = i[1]
        self.cities_mapping = cities_mapping

    def run_overall_ga(self):
        possible_solutions = m.factorial(self.number_of_cities - 1)
        ninitpop = self.initial_pop_size
        ## create an empty dataframe to store the solutions
        all_solutions_generated = pd.DataFrame(columns=["Route", "Cost"])
        """start a for loop and run the whole process for mentioned number
         of times"""
        print("Starting {} iterations of the GA".format(self.noverall))
        # generating initial population
        """We only generate a population initially, after  the first run ,
		we take the best solutions from the previous run and continue with
         the process"""
        if all_solutions_generated.shape[0] == 0:
            initial_pop_cost = self.initial_pop_cost()
            sorted_init_pop = initial_pop_cost.sort_values("Cost")
        else:
            sorted_init_pop = all_solutions_generated.head(self.initial_pop_size)

        # selecting the elite few
        elite_few_df = self.the_elite_few()
        best_sol = []
        """Generating a random number based on which we either
         mutate or do a crossover"""
        mating_factor = np.random.uniform(
            0, 1, 1
        )  # Random pick to decide on Mutation / crossover

        for i in range(self.noverall):
            if mating_factor < 0.2:
                mutated_population_wth_cost = self.mutation_function(
                    all_solutions_generated
                )
                all_solutions_generated.append(mutated_population_wth_cost)
            else:
                crossover_population = self.routes_after_cross_over()
                all_solutions_generated = all_solutions_generated.append(
                    crossover_population
                )

            all_solutions_generated.Route = all_solutions_generated.Route.map(
                lambda x: tuple(x)
            )
            unique_sols_generated = (
                all_solutions_generated.drop_duplicates().sort_values("Cost")
            )
            all_solutions_generated = all_solutions_generated.sort_values("Cost").head(
                ninitpop
            )  # only take the top ninitpop number of solutions
            all_solutions_generated.Route = all_solutions_generated.Route.map(
                lambda x: list(x)
            )

        print(
            "-----------------------------------------------------------------"
        )
        print(
            """Best solution for initial population size of {} and number 
            of runs {} is \n {}""".format(
                self.initial_pop_size,
                self.noverall,
                all_solutions_generated.sort_values("Cost").head(1),
            )
        )
        print(
            "Generated {}({}%) of the {} solutions".format(
                all_solutions_generated.shape[0],
        np.round((len(all_solutions_generated) / possible_solutions) * 100,3),
                possible_solutions,
            )
        )
        final_sol = all_solutions_generated.sort_values("Cost").head(1)

        #### Final Solution ( Cities)
        final_route = []
        starting_point = self.dist_mat.index[0]
        for i in final_sol.Route.values[0]:
            final_route.append(self.cities_mapping[i])
        final_route = [starting_point] + final_route + [starting_point]
        total_cost = ray.get(self.fitness_function.remote(self,list(final_sol.Route.values[0])))

        print(
            """Total distance travelled to cover the final route of \n {} is {} KM.
             (Generated from initial population size of {})""".format(
                " => ".join(final_route), total_cost, self.initial_pop_size
            )
        )
        print(
            "-----------------------------------------------------------------"
        )
        best_sol.append(final_sol)

        return (final_sol,total_cost)


if __name__ != "__main__":
    try:
        from geopy.geocoders import Nominatim
    except ModuleNotFoundError:
        warnings.warn(
            """Could not load geopy,
			the package is either not available for the os you are using or
            is not available in the scope of the class.This will not affect
            the functionality of the class,
			you cannot generate plots
            """,
            stacklevel=1,
        )
    try:
        import folium
    except ModuleNotFoundError:
        warnings.warn(
            """Could not load folium,
			the package is either not available for the os you are using or
            is not availablein the scope of the class.This will not affect
            the functionality of the class,
            you cannot generate plots""",
            stacklevel=1,
        )
    try:
        from branca.element import Figure
    except ModuleNotFoundError:
        warnings.warn(
            """Could not load branca,
            the package is either not available for the os you are using or
            is not available in the scope of the class.This will not affect
            the functionality of the class,
            you cannot generate plots""",
            stacklevel=1,
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    from numpy import math as m
