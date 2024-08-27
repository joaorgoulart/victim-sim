import random
import numpy as np

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, elitism_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate

    def initialize_population(self, victims):
        return [random.sample(victims, len(victims)) for _ in range(self.population_size)]

    def fitness(self, sequence, victims_data, start_pos, end_pos):
        total_distance = 0
        total_gravity = 0
        current_pos = start_pos

        for victim_id in sequence:
            victim_pos = victims_data[victim_id]['position']
            total_distance += self.calculate_distance(current_pos, victim_pos)
            total_gravity += victims_data[victim_id]['gravity']
            current_pos = victim_pos

        total_distance += self.calculate_distance(current_pos, end_pos)
        
        if total_distance == 0:
            return 0
        return total_gravity / total_distance

    def calculate_distance(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)

    def select_parents(self, population, fitnesses):
        return random.choices(population, weights=fitnesses, k=2)

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            
            crossover_point = random.randint(1, len(parent1))
            child = parent1[:crossover_point] + [gene for gene in parent2 if gene not in parent1[:crossover_point]]
            return child
        else:
            return parent1

    def mutate(self, sequence):
        for i in range(len(sequence)):
            if random.random() < self.mutation_rate:
                j = random.randint(0, len(sequence) - 1)
                sequence[i], sequence[j] = sequence[j], sequence[i]
        return sequence

    def evolve(self, population, victims_data, start_pos, end_pos):
        fitnesses = [self.fitness(seq, victims_data, start_pos, end_pos) for seq in population]
        
        elite_size = int(self.elitism_rate * self.population_size)
        elite = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:elite_size]
        
        new_population = [ind for ind, _ in elite]
        
        while len(new_population) < self.population_size:
            parents = self.select_parents(population, fitnesses)
            child = self.crossover(parents[0], parents[1])
            child = self.mutate(child)
            new_population.append(child)
        
        return new_population

    def run(self, victims, victims_data, start_pos, end_pos, generations):
        population = self.initialize_population(victims)
        
        for _ in range(generations):
            population = self.evolve(population, victims_data, start_pos, end_pos)
        
        best_sequence = max(population, key=lambda seq: self.fitness(seq, victims_data, start_pos, end_pos))
        return best_sequence