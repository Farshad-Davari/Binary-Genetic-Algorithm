from numpy.random import rand, randint
import numpy as np
import matplotlib.pyplot as plt

# Stage 1 : Parameters
bounds = [[-3, 12.1], [4.1, 5.8]]
iteration = 100
bits = 20 
popSize = 100
crossoverRate = 0.8
mutationRate = 0.01

# Stage 2 : Initial Population
pop = [randint(0, 2, bits*len(bounds)).tolist() for _ in range(popSize)]

# Objective Function
def objectiveFunction(I):
  x = I[0]
  y = I[1]
  objectiveMin = 21.5 + x*np.sin(4*np.pi*x) + y*np.sin(20*np.pi*y)
  objectiveMax = 1 / (1 + objectiveMin)
  return objectiveMax

# Stage 3 : Decoding
def decoding(bounds, bits, chromosome):
  realChromosome = list() #empty sequence
  for i in range(len(bounds)):
    st, en = i * bits, (i*bits) + bits #extract the chromosome
    sub = chromosome[st:en]
    chars = ''.join([str(s) for s in sub]) #convert to chars
    integer = int(chars, 2) #convert to integer
    realValue = bounds[i][0] + (integer / (2**bits)) * (bounds[i][1] - bounds[i][0])
    realChromosome.append(realValue)
  return realChromosome

# Stage 4 and 5 : Evaluation and Selection
def selection(pop, fitness, popSize):
  nextGeneration = list() #empty sequence
  elite = np.argmax(fitness) #argmax => return indices of maximum value along an axis
  nextGeneration.append(pop[elite]) #keep the best
  P = [f / sum(fitness) for f in fitness]
  index = list(range(int(len(pop))))
  indexSelected = np.random.choice(index, size = popSize - 1, replace = False, p = P) #choice => generate a random sample of a given 1-d array
  s = 0
  for j in range(popSize - 1):
    nextGeneration.append(pop[indexSelected[s]])
    s += 1

  return nextGeneration
  
# Stage 6 : Crossover
def crossover(pop, crossoverRate):
  offspring = list() #empty sequence
  for i in range(int(len(pop) / 2)):
    p1 = pop[2*i - 1].copy()
    p2 = pop[2*i].copy()
    if rand() < crossoverRate:
      cp = randint(1, len(p1) - 1, size = 2)   #tow random cutting points
      while cp[0] == cp[1]:
        cp = randint(1, len(p1) - 1, size = 2) #tow random cutting points

      cp = sorted(cp)
      c1 = p1[:cp[0]] + p2[cp[0]:cp[1]] + p1[cp[1]:]  
      c2 = p2[:cp[0]] + p1[cp[0]:cp[1]] + p2[cp[1]:] 

      offspring.append(c1)
      offspring.append(c2)

    else:
      offspring.append(p1)  
      offspring.append(p2)  

  return offspring

# Stage 7 : Mutation
def mutation(pop, mutationRate):
  offspring = list() #empty sequence
  for i in range(int(len(pop))):
    p1 = pop[i].copy() #parent
    if rand() < mutationRate:
      cp = randint(0, len(p1)) #random gene
      c1 = p1
      if c1[cp] == 1:
        c1[cp] = 0 #flip
      else:
        c1[cp] = 1

      offspring.append(c1)
    else:
      offspring.append(p1)      

  return offspring

# Main Program
bestFitness = []
for gen in range(iteration):
  offspring = crossover(pop, crossoverRate)
  offspring = mutation(pop, mutationRate)

  for s in offspring:
    pop.append(s)


  realChromosome = [decoding(bounds, bits, p) for p in pop]
  fitness = [objectiveFunction(d) for d in realChromosome] #fitness value

  index = np.argmax(fitness)
  currentBest = pop[index]
  bestFitness.append(1 / max(fitness) - 1)
  pop = selection(pop, fitness, popSize)

fig = plt.figure()
plt.plot(bestFitness)
fig.suptitle("Evolution of the best chromosome")
plt.xlabel("Iteration")
plt.ylabel("Objective function value")
plt.show()
print("Max objective function value: ", max(bestFitness))
print("Optimal solution", decoding(bounds, bits, currentBest))