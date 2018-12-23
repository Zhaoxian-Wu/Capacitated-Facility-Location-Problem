import math
import pickle
import random
import re
import time

import matplotlib.pyplot as plt

facilityCount, customorCount = 0, 0
assignCost = [[0.0]]
openCost = [0.0]
capacity = [0.0]
openCost = [0.0]
demand = [0.0]

def cost(assign:[int]):
  isOpen = [False] * facilityCount
  for i in range(customorCount):
    isOpen[assign[i]] = True

  _cost = 0
  for i in range(customorCount):
    _cost += assignCost[i][assign[i]]
  for i in range(facilityCount):
    if isOpen[i]:
      _cost += openCost[i]
  return _cost

def assignRepair(assign:[float]):
  # 工厂的负载
  load = [0] * facilityCount
  # 被分配到某工厂的用户
  assigned = [[] for _ in range(facilityCount)]
  # 超载的工厂序号
  overload = set()
  for c in range(customorCount):
    fac = assign[c] 
    assigned[fac].append(c)
    load[fac] += demand[c]
    if load[fac] > capacity[fac]:
      # 过载
      overload.add(fac)
  while(len(overload) != 0):
    facility = random.sample(overload, 1)[0]
    customor = random.choice(assigned[facility])
    # 移出
    assert(assign[customor] == facility)
    oldFac = assign[customor]
    assigned[oldFac].remove(customor)
    load[oldFac] -= demand[customor]
    if load[oldFac] <= capacity[oldFac]:
      overload.remove(oldFac)
    
    # 重新分配
    newFac = random.randint(0, facilityCount - 1)
    assigned[newFac].append(customor)
    load[newFac] += demand[customor]
    if load[newFac] > capacity[newFac]:
      overload.add(newFac)
    assign[customor] = newFac

  return assign
  
# ============================================================
# SA
changeProp = 0.1
T0 = 10000
factor = 0.999
STOP_TEMPERATURE = 1

def assignNeighbor(assign:[float]):
  # 工厂被分配的用户数
  facilities = [0] * facilityCount

  for i in range(customorCount):
    facilities[assign[i]] += 1
    
  res = assign.copy()

  for i in range(customorCount):
    if random.random() < 2/facilities[assign[i]]:
      res[i] = random.randint(0, facilityCount-1)

  return res

def assignInit():
  assign = [facilityCount] * customorCount
  
  leaveCapacity = capacity.copy()
  order = list(range(customorCount))
  try:
    random.shuffle(order)
  except RecursionError as e:
    print(order)
    raise(e)
  for customor in order:
    bestSuit = facilityCount
    minLeave = 5000000
    for fac in range(facilityCount):
      leave = leaveCapacity[fac]
      if leave >= demand[customor] and leave < minLeave:
        bestSuit = fac
        minLeave = leave
    if bestSuit == facilityCount:
      # 不能放下,另起炉灶
      return assignInit()
    assign[customor] = bestSuit
    leaveCapacity[bestSuit] -= demand[customor]
  return assign

# ============================================================
# GA
k_tournament = 10
reproduceProp = 0.5
inheritanceProp = 0.5
POPULATION_SIZE = 400

def crossover(parent1:[float], parent2:[float]):
  # 随机交叉
  # customorCount = len(parent1)
  # changeCount = math.ceil(customorCount*changeProp)
  # o1, o2 = parent1.copy(), parent2.copy()
  # for i in random.choices(range(customorCount), k=changeCount):
  #   o1[i], o2[i] = o2[i], o1[i]
  # return o1, o2

  # ==========================================================
  # 按块遗传
  # 被分配到对应工厂的用户列表
  facilities_1 = [[] for _ in range(facilityCount)]
  facilities_2 = [[] for _ in range(facilityCount)]

  # 有被分配的工厂序号
  wasAssign_1 = set()
  wasAssign_2 = set()
  for i in range(customorCount):
    facilities_1[parent1[i]].append(i)
    wasAssign_1.add(parent1[i])
    facilities_2[parent2[i]].append(i)
    wasAssign_2.add(parent2[i])
    
  # 抽出几个有被分配用户的工厂
  chromosomes_1 \
    = random.sample(wasAssign_1, k=math.ceil(len(wasAssign_1)*inheritanceProp))
  chromosomes_2 \
    = random.sample(wasAssign_2, k=math.ceil(len(wasAssign_2)*inheritanceProp))
  # 子代1
  o1 = [facilityCount] * customorCount
  o2 = [facilityCount] * customorCount
  for fac in chromosomes_1:
    for cus in facilities_1[fac]:
      o1[cus] = fac
  for fac in chromosomes_2:
    for cus in facilities_2[fac]:
      o2[cus] = fac
  # 补全未被分配的维度
  for i in range(customorCount):
    if o1[i] == facilityCount:
      o1[i] = parent2[i]
  for i in range(customorCount):
    if o2[i] == facilityCount:
      o2[i] = parent1[i]
  return o1, o2
  # ==========================================================
  
def mutation(individual):
  res = individual.copy()
  # 随机改变一段
  changeCount = math.ceil(len(individual)*changeProp)
  end = random.randint(0, len(individual) - 1)
  for i in range(end, end-changeCount, -1):
    res[i] = random.randint(0, facilityCount-1)
  return res

mutation2 = assignNeighbor

def naturalSelection(population, newIndividuals):
  # k-锦标赛
  for individual in newIndividuals:
    population.append(individual)
    competitors = random.choices(range(len(population)), k=k_tournament)
    loser = max(competitors, key=lambda i: cost(population[i]))
    population.pop(loser)

# ============================================================
def SA():
  global T0, factor, STOP_TEMPERATURE

  T = T0
  
  assign = assignInit()
  argmin = assign
  value = cost(argmin)

  record = []

  while(T > STOP_TEMPERATURE):
    for _ in range(facilityCount**2):
      neigh = assignNeighbor(assign)
      neigh = assignRepair(neigh)
      newValue = cost(neigh)
      if newValue < value:
        value = newValue
        assign = neigh
      elif(math.exp(-(newValue-value)/T) >= random.random()):
        assign = neigh
    T *= factor
    record.append(value)
  
  recordResult({
    'type': 'SA',
    'date': time.time(),
    'changeProp': changeProp,
    'T0': T0,
    'factor': factor,
    'STOP_TEMPERATURE': STOP_TEMPERATURE,
    'x-axis': [i for i in range(len(record))],
    'y-axis': record,
  })

  return argmin
def GA():
  record = []
  
  population = \
    [assignInit() for _ in range(POPULATION_SIZE)]
  population = [assignRepair(e) for e in population]
  
  argmin = min(population, key=cost)
  value = cost(argmin)

  GENERATION = 80
  reproduceCount = math.ceil(POPULATION_SIZE * reproduceProp)

  for _ in range(GENERATION):
    newOffspring = []
    for _ in range(reproduceCount):
      p1, p2 = random.choices(population, k=2)
      offspring1, offspring2 = crossover(p1, p2)

      offspring3 = mutation(offspring1)
      offspring4 = mutation(offspring2)
      offspring5 = mutation2(offspring1)
      offspring6 = mutation2(offspring2)

      newOffspring.append(offspring1)
      newOffspring.append(offspring2)
      newOffspring.append(offspring3)
      newOffspring.append(offspring4)
      newOffspring.append(offspring5)
      newOffspring.append(offspring6)
    # 修正非法解
    newOffspring = [assignRepair(o) for o in newOffspring]
    
    naturalSelection(population, newOffspring)

    newAssign = min(newOffspring, key=cost)
    newValue = cost(newAssign)
    if newValue < value:
      value = newValue
      argmin = newAssign
    record.append(value)
  recordResult({
    'type': 'GA',
    'date': time.time(),
    'population': POPULATION_SIZE,
    'genaration': GENERATION,
    'k_tournament': k_tournament,
    'changeProp': changeProp,
    'inheritanceProp': inheritanceProp,
    'reproduceProp': reproduceProp,
    'x-axis': [i for i in range(len(record))],
    'y-axis': record,
  })
  return argmin
    
def loadFile(file):
  global facilityCount, customorCount, capacity, openCost, assignCost, demand
  with open(file, 'r') as f:
    line = f.readline()
    facilityCount, customorCount = [int(p) for p in line.split()]
    capacity = [0] * facilityCount
    openCost = [0] * facilityCount
    for i in range(facilityCount):
      line = f.readline()
      capacity[i], openCost[i] = [float(p) for p in line.split()]
    # 处理空格和换行
    buffer = [float(p) for p in ''.join(f.readlines()).split() if p[0] != '\x00']
    assert(len(buffer) == facilityCount*customorCount + customorCount)
    
    # 客户需要
    demand = buffer[:customorCount]
    # 分配花费
    assignCost = [[0] * facilityCount for _ in range(customorCount)]
    for i in range(customorCount):
      assignCost[i] = buffer[\
         customorCount + i * facilityCount \
        :customorCount + (i+1) * facilityCount \
      ]
  assert(len(assignCost) == customorCount)
  assert(len(assignCost[0]) == facilityCount)

def recordResult(record):
  print('正在写入数据')
  fileName = '.drawData'
  result = None
  with open(fileName, 'r+b') as f:
    result = pickle.load(f)
    result.append(record)
    f.seek(0, 0)
    f.truncate()
    pickle.dump(result, f)
  print('写入完成')

def runCal():
  
  table = ''
  table += '||Result|Time(s)|\n'
  table += '|:-:|:-:|:-:|\n'
  detail = ''
  for i in range(1, 72):
    print('正在计算:p{}'.format(i))
  
    inFile = 'Instances/p{}'.format(i)
    loadFile(inFile)

    beg = time.time()
    assign = GA()
    # assign = SA()

    if not isValid(assign):
      print('[{}] invalid output')
    
    timecost = time.time() - beg
    result = cost(assign)
    table += 'p{}|{}|{:.2f}\n'.format(i, result, timecost)

    detail += '**p{}**  \n{}  \n'.format(i, result)
    openFac = [False] * facilityCount
    for fac in assign:
      openFac[fac] = True
    for o in openFac:
      if o:
        detail += '1 '
      else:
        detail += '0 '
    detail += '  \n'
    for fac in assign:
      detail += str(fac) + ' '
    detail += '\n\n---\n'
  with open('result.md', 'w') as f:
    f.write('# result table\n')
    f.write(table)
    f.write('# detail solution\n')
    f.write(detail)
    
def isValid(assign):
  load = [0] * facilityCount
  for i in range(customorCount):
    fac = assign[i]
    load[fac] += demand[i]
    if load[fac] > capacity[fac]:
      return False
  return True

if __name__== '__main__':
  runCal()
