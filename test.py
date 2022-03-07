import math

def objective_function(I):
    x = I[0]
    y = I[1]
    Objective_min = 21.5 + x*math.sin(4*math.pi*x) + y*math.sin(20*math.pi*y)  
    Objective_max = 1/(1 + Objective_min) # 
    
    return Objective_max


bounds = [[-10, 10], [-10, 10]]
iteration = 200
bits = 20 
pop_size = 100
crossover_rate = 0.8
mutation_rate = 0.2

def crossover(pop, crossover_rate):
    offspring = list()
    for i in range(int(len(pop)/2)):
        p1 = pop[2*i-1].copy() # parent 1
        p2 = pop[2*i].copy() # parent 2   
    
    return offspring


def mutation(pop, mutation_rate):
    offspring = list()
    for i in range(int(len(pop))):
    
    


def selection(pop, fitness, pop_size):
    next_generation = list()
    elite = np.argmax(fitness)
    next_generation.append(pop[elite])  


    return next_generation


def decoding(bounds, bits, chromosome):
	real_chromosome = list()
	for i in range(len(bounds)):
    



pop = [randint(0, 2, bits*len(bounds)).tolist() for _ in range(pop_size)]

# main program
best_fitness = []
for gen in range(iteration):