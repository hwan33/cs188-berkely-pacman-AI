# myTeam.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
import distanceCalculator
#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'DummyAgent', second = 'DummyAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """

  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  Things to consider:
  -location of food pellets we want to eat AND also of those we want to defend
  -what team we're on
  -what side of the board we're on
  -location of our opponents
  -location of capsules (to make enemies scared)
  -location of our own capsules (that will make us scared)
  -if the opponents are scared
  -if we are scared dont approach invaders
  -don't care how close invaders are to food, only how close they are to getting back to their side
  -if we have eaten more than 5 capsules always return to side
  """
  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).
    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)
    IMPORTANT: This method may run for at most 15 seconds.
    """
    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    CaptureAgent.registerInitialState(self, gameState)
    '''
    Your initialization code goes here, if you need any.
    '''
    self.start = gameState.getAgentPosition(self.index)
  def chooseAction(self, gameState):
    acts = gameState.getLegalActions(self.index)
    values = [self.evaluate(gameState, a) for a in acts]
    max_val = max(values)
    act_list = [a for a, v in zip(acts, values) if v == max_val]
    food_num = len(self.getFood(gameState).asList())
    if food_num <= 2:
      best_dist = 10000
      for action in acts:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start,pos2)
        if dist < best_dist:
          best_act = action
          best_dist = dist
      return best_act
    return random.choice(act_list)
  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != util.nearestPoint(pos):
      return successor.generateSuccessor(self.index, action)
    else:
      return successor
  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    successor = self.getSuccessor(gameState, action)
    state_agent1 = successor.getAgentState(self.index)
    enemy = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    team = [successor.getAgentState(i) for i in self.getTeam(successor)]
    team.remove(state_agent1)
    enemy_pacman = [a for a in enemy if a.isPacman and a.getPosition() != None]
    enemy_ghost = [a for a in enemy if not a.isPacman and a.getPosition() != None]
    defense = 1.0 * len(enemy_pacman)/len(enemy)
    closest_food_distance = float("+inf")
    food_we_are_defending = self.getFoodYouAreDefending(gameState)
    for invader in enemy_pacman:
      for rows in range(food_we_are_defending.height):
        for cols in range(food_we_are_defending.width):
          if food_we_are_defending[cols][rows]:
            distance_invasion = self.getMazeDistance(invader.getPosition(), (cols, rows))
            if (distance_invasion < closest_food_distance):
              defense += 1/(distance_invasion + 1)
              closest_food_distance = distance_invasion
      defense += 1/self.getMazeDistance(state_agent1.getPosition(), invader.getPosition())
      defense += 1/(food_we_are_defending.width - invader.getPosition()[0] + 1)
    for non_invader in enemy_ghost:
      defense -= 1/self.getMazeDistance(state_agent1.getPosition(), non_invader.getPosition())
    # if (food_we_are_defending.width - us[0].getPosition()[0]) > 0:
    #   defense -= 0.1
    if (successor.getAgentState(self.index).numCarrying > 3):
      defense = 1
    if (defense >= 0.5):
      features = self.getDefensiveFeatures(gameState, action)
      weights = self.getDefensiveWeights(gameState, action)
      self.model = 0
    else:
      features = self.getOffensiveFeatures(gameState, action)
      weights = self.getOffensiveWeights(gameState, action)
      self.model = 1
    return features * weights
  def getDefensiveFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    state_agent1 = successor.getAgentState(self.index)
    pos_agent1 = state_agent1.getPosition()
    team = [successor.getAgentState(i) for i in self.getTeam(successor)]
    team.remove(state_agent1)
    partnerPos = team[0].getPosition()
    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if state_agent1.isPacman: features['onDefense'] = 0
    # Computes distance to invaders we can see
    enemy = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemy_pacman = [a for a in enemy if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(enemy_pacman)
    if len(enemy_pacman) > 0:
      dists = [self.getMazeDistance(pos_agent1, a.getPosition()) for a in enemy_pacman]
      features['invaderDistance'] = min(dists)
    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1
    return features
  def getDefensiveWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10000, 'stop': -100, 'reverse': -2}
  def getOffensiveFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    state_agent1 = successor.getAgentState(self.index)
    pos_agent1 = state_agent1.getPosition()
    team = [successor.getAgentState(i) for i in self.getTeam(successor)]
    team.remove(state_agent1)
    partnerPos = team[0].getPosition()
    enemy = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    enemy_ghost = [a for a in enemy if not a.isPacman and a.getPosition() != None]
    # Compute distance to the nearest food
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      minDistance = min([self.getMazeDistance(pos_agent1, food) for food in foodList])
      features['distanceToFood'] = minDistance
    # Compute distance to enemy capsule
    if self.red:myCapsules = gameState.getBlueCapsules()
    else: myCapsules = gameState.getRedCapsules()
    if len(myCapsules) > 0:
      pos_agent1 = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(pos_agent1, capsule) for capsule in myCapsules])
      features['distanceToFood'] = minDistance
    features['distanceFromPartner'] = self.getMazeDistance(pos_agent1, partnerPos)
    features['distanceToNonInvaders'] = min([self.getMazeDistance(pos_agent1, enemy.getPosition()) for enemy in enemy_ghost])
    return features
  def getOffensiveWeights(self, gameState, action):
    successor = self.getSuccessor(gameState, action)
    state_agent1 = successor.getAgentState(self.index)
    team = [successor.getAgentState(i) for i in self.getTeam(successor)]
    team.remove(state_agent1)
    food_we_are_defending = self.getFoodYouAreDefending(gameState)
    if (food_we_are_defending.width - team[0].getPosition()[0]) > 0:
      weight = 0
    else:
      weight = 10
    return {'successorScore': 100, 'distanceToFood': -1, 'distanceToCapsule': 50, 'distanceFromPartner': weight, 'distanceToNonInvaders': weight * 25}
