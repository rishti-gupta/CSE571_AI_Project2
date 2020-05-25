# multiAgents.py
# --------------
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


from util import manhattanDistance
from game import Directions
import random, util

from game import Agent

class ReflexAgent(Agent):
    """
      A reflex agent chooses an action at each choice point by examining
      its alternatives via a state evaluation function.

      The code below is provided as a guide.  You are welcome to change
      it in any way you see fit, so long as you don't touch our method
      headers.
    """


    def getAction(self, gameState):
        """
        You do not need to change this method, but you're welcome to.

        getAction chooses among the best options according to the evaluation function.

        Just like in the previous project, getAction takes a GameState and returns
        some Directions.X for some X in the set {North, South, West, East, Stop}
        """
        # Collect legal moves and successor states
        legalMoves = gameState.getLegalActions()

        # Choose one of the best actions
        scores = [self.evaluationFunction(gameState, action) for action in legalMoves]
        bestScore = max(scores)
        bestIndices = [index for index in range(len(scores)) if scores[index] == bestScore]
        chosenIndex = random.choice(bestIndices) # Pick randomly among the best

        "Add more of your code here if you want to"

        return legalMoves[chosenIndex]

    def evaluationFunction(self, currentGameState, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (pacman.py) and returns a number, where higher numbers are better.

        The code below extracts some useful information from the state, like the
        remaining food (newFood) and Pacman position after moving (newPos).
        newScaredTimes holds the number of moves that each ghost will remain
        scared because of Pacman having eaten a power pellet.

        Print out these variables to see what you're getting, then combine them
        to create a masterful evaluation function.
        """
        # Useful information you can extract from a GameState (pacman.py)
        successorGameState = currentGameState.generatePacmanSuccessor(action)
        newPos = successorGameState.getPacmanPosition()
        newFood = successorGameState.getFood()
        newGhostStates = successorGameState.getGhostStates()
        newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]

        "*** YOUR CODE HERE ***"
        if successorGameState.isWin():
            return float("inf")

        # if the distance between the ghost and pacman is less than 2, the pacman looses
        for state in newGhostStates:
            if util.manhattanDistance(state.getPosition(), newPos) < 2:
                return float("-inf")

        # eat the food if the food is already scared
        if currentGameState.getNumFood() > successorGameState.getNumFood():
            return float("inf")

        minDist = float("inf")
        foodList = newFood.asList();

        # reciprocal of 'distance to food' as features
        for food in list(foodList):
            dist = util.manhattanDistance(newPos, food)
            minDist = min(minDist, dist)

        return 1.0/minDist

def scoreEvaluationFunction(currentGameState):
    """
      This default evaluation function just returns the score of the state.
      The score is the same one displayed in the Pacman GUI.

      This evaluation function is meant for use with adversarial search agents
      (not reflex agents).
    """
    return currentGameState.getScore()

class MultiAgentSearchAgent(Agent):
    """
      This class provides some common elements to all of your
      multi-agent searchers.  Any methods defined here will be available
      to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

      You *do not* need to make any changes here, but you can if you want to
      add functionality to all your adversarial search agents.  Please do not
      remove anything, however.

      Note: this is an abstract class: one that should not be instantiated.  It's
      only partially specified, and designed to be extended.  Agent (game.py)
      is another abstract class.
    """

    def __init__(self, evalFn = 'scoreEvaluationFunction', depth = '2'):
        self.index = 0 # Pacman is always agent index 0
        self.evaluationFunction = util.lookup(evalFn, globals())
        self.depth = int(depth)

class MinimaxAgent(MultiAgentSearchAgent):
    """
      Your minimax agent (question 2)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action from the current gameState using self.depth
          and self.evaluationFunction.

          Here are some method calls that might be useful when implementing minimax.

          gameState.getLegalActions(agentIndex):
            Returns a list of legal actions for an agent
            agentIndex=0 means Pacman, ghosts are >= 1

          gameState.generateSuccessor(agentIndex, action):
            Returns the successor game state after an agent takes an action

          gameState.getNumAgents():
            Returns the total number of agents in the game
        """
        "*** YOUR CODE HERE ***"
        score = float("-inf")
        action = Directions.STOP
        # driver code
        for legal_action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, legal_action)
            nextScore = self.helper(nextState, 0, 1)
            if nextScore > score:
                score = nextScore
                action = legal_action
        return action

    # defining value function
    def helper(self, state, depth, agent_index):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent_index == 0:
            return self.maxValue(state, depth)
        else:
            return self.minValue(state, depth, agent_index)

    # defining the minimum score function
    def minValue(self, state, depth, agent_index):
        minScore = float("inf")
        ghostNum = state.getNumAgents() - 1
        for action in state.getLegalActions(agent_index):
            if agent_index == ghostNum:
                minScore = min(minScore, self.helper(state.generateSuccessor(agent_index, action), depth + 1, 0))
            else:
                minScore = min(minScore,
                               self.helper(state.generateSuccessor(agent_index, action), depth, agent_index + 1))
        return minScore

    # defining the maximum score function
    def maxValue(self, state, depth):
        maxScore = float("-inf")
        for action in state.getLegalActions(0):
            maxScore = max(maxScore, self.helper(state.generateSuccessor(0, action), depth, 1))
        return maxScore

class AlphaBetaAgent(MultiAgentSearchAgent):
    """
      Your minimax agent with alpha-beta pruning (question 3)
    """

    def getAction(self, gameState):
        """
          Returns the minimax action using self.depth and self.evaluationFunction
        """
        "*** YOUR CODE HERE ***"

        score = float("-inf")
        alpha = float("-inf")
        beta = float("inf")
        action = Directions.STOP

        # driver code
        for legal_action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, legal_action)
            nextScore = self.helper(nextState, 0, 1, alpha, beta)
            if nextScore > score:
                score = nextScore
                action = legal_action
            alpha = max(alpha, score)
        return action

    # defining the value function
    def helper(self, state, depth, agent_index, alpha, beta):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent_index == 0:
            return self.maxValue(state, depth, alpha, beta)
        else:
            return self.minValue(state, depth, agent_index, alpha, beta)

    # defining maximum score function
    def maxValue(self, state, depth, alpha, beta):
        maxScore = float("-inf")
        for action in state.getLegalActions(0):
            maxScore = max(maxScore, self.helper(state.generateSuccessor(0, action), depth, 1, alpha, beta))

            if maxScore > beta:
                return maxScore
            alpha = max(alpha, maxScore)
        return maxScore

    # defining minimum score function
    def minValue(self, state, depth, agent_index, alpha, beta):
        minScore = float("inf")
        ghostNum = state.getNumAgents() - 1
        for action in state.getLegalActions(agent_index):
            if agent_index == ghostNum:
                minScore = min(minScore, self.helper(state.generateSuccessor(agent_index, action), depth + 1, 0, alpha, beta))
            else:
                minScore = min(minScore, self.helper(state.generateSuccessor(agent_index, action), depth, agent_index + 1, alpha, beta))

            if minScore < alpha:
                return minScore
            beta = min(beta, minScore)
        return minScore

class ExpectimaxAgent(MultiAgentSearchAgent):
    """
      Your expectimax agent (question 4)
    """

    def getAction(self, gameState):
        """
          Returns the expectimax action using self.depth and self.evaluationFunction

          All ghosts should be modeled as choosing uniformly at random from their
          legal moves.
        """
        "*** YOUR CODE HERE ***"
        score = float("-inf")
        action = Directions.STOP

        # driver code
        for legal_action in gameState.getLegalActions(0):
            nextState = gameState.generateSuccessor(0, legal_action)
            nextScore = self.helper(nextState, 0, 1)
            if nextScore > score:
                score = nextScore
                action = legal_action
        return action

    # defining value function
    def helper(self, state, depth, agent_index):
        if depth == self.depth or state.isWin() or state.isLose():
            return self.evaluationFunction(state)
        elif agent_index == 0:
            return self.maxValue(state, depth)
        else:
            return self.expectedValue(state, depth, agent_index)

    # defining maximum score function
    def maxValue(self, state, depth):
        maxScore = float("-inf")
        for action in state.getLegalActions(0):
            maxScore = max(maxScore, self.helper(state.generateSuccessor(0, action), depth, 1))
        return maxScore

    # defining average(expected) score function insteaad of minimum score function
    def expectedValue(self, state, depth, agent_index):
        averageScore = 0
        ghostNum = state.getNumAgents() - 1
        for action in state.getLegalActions(agent_index):
            if agent_index == ghostNum:
                averageScore += self.helper(state.generateSuccessor(agent_index, action), depth + 1, 0)
            else:
                averageScore += self.helper(state.generateSuccessor(agent_index, action), depth, agent_index + 1)
        return averageScore

def betterEvaluationFunction(currentGameState):
    """
      Your extreme ghost-hunting, pellet-nabbing, food-gobbling, unstoppable
      evaluation function (question 5).

      DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    pacPos = currentGameState.getPacmanPosition();

    if currentGameState.isWin():
        return float("inf")

    if currentGameState.isLose():
        return float("-inf")

    foodList = currentGameState.getFood().asList()
    foodDist = []

    # finding minimum distance to food from current location of the pacman
    for food in foodList:
        foodDist.append(util.manhattanDistance(pacPos, food))
    minFoodDist = min(foodDist)

    ghostStates = currentGameState.getGhostStates()
    ghostAlive = []
    ghostDead = []

    # separate ghosts as dead or alive
    # scared ghosts are the dead ones which is better for pacman to eat to gain more points
    # the pacman has to still run away from the alive ghosts which are the active ones
    for ghost in ghostStates:
        if ghost.scaredTimer:
            ghostDead.append(ghost)
        else:
            ghostAlive.append(ghost)

    ghostAliveDist = 0
    ghostDeadDist = 0
    # finding distance to the alive ghost from current location of the pacman
    if ghostAlive:
        ghostAliveDist = min(map(lambda x: util.manhattanDistance(pacPos, x.getPosition()), ghostAlive))
    else:
        ghostAliveDist = float("inf")
        ghostAliveDist = max(ghostAliveDist, 10)

    if ghostDead:
        ghostDeadDist = min(map(lambda x: util.manhattanDistance(pacPos, x.getPosition()), ghostDead))
    else:
        ghostDeadDist = 0

    # the values to calculate the score of the game are chosen by hit and trial (arbitrary)
    # pacman must eat the food dots. Thus, -15 score has been given to it as it is bad for the pacman to ignore any dot.
    sc_food = -15* minFoodDist
    # we want the distance to alive ghost feature to be more important than food. Thus, the pacman should not eat the food
    # dot if an alive ghost is there. Thus, discourage that behaviour by providing the reciprocal of the distance
    # to the alive ghost and multiplying it with -20.
    sc_ghostAlive = -20* (1.0/ghostAliveDist)

    # it is good for the pacman to eat the dead ghost for more points. Thus, it is multiplied by -20, which is more than the food
    # dot as we encourage our pacman to eat the dead ghost over food
    sc_ghostDead = -20 * ghostDeadDist

    # penalty on any capsule that is remaining
    penalty_capsule = -200* len(currentGameState.getCapsules())

    # penalty on any food dot that is remaining
    penalty_foodRemaining = -50* len(foodList)

    # calculating the final score
    score = scoreEvaluationFunction(currentGameState) + sc_food + sc_ghostAlive + sc_ghostDead + penalty_capsule + penalty_foodRemaining

    return score

# Abbreviation
better = betterEvaluationFunction

