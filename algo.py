import numpy as np #import numpy library
import dataset     # import the dataset code which gives us all the information of the data


class LinUCB:
    """
    Implementing LinUCB algorithm
    """

    def __init__(self, alpha, context="user"):
        """
        Parameters passed:
        ----------
        alpha :LinUCB parameter
        context: 'user' or 'both'(item+user): to decide the feature vector
        """
        self.n_features = len(dataset.a_features[0]) #length of feature vector for an article
        if context == "user":
            self.context = 1
        elif context == "both":
            self.context = 2
            self.n_features *= 2 #we use 12 dimensional feature vector in case of both user and articles

        self.A = np.array([np.identity(self.n_features)] * dataset.n_arms) #collection of identity matrices of size=n_features for each article
        self.A_inv = np.array([np.identity(self.n_features)] * dataset.n_arms) # same collection as A
        self.b = np.zeros((dataset.n_arms, self.n_features, 1)) # a matrix where each row corresponds to an article ,and b for each article is of n_features dimension
        self.alpha = round(alpha, 1) # alpha rounded off to one decimal place
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ", context:" + context + ")" # to represnt which algorithm we are dealing with and its parameters

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t : number of trial
        user : user features
        pool_idx :pool indexes for article identification
        """

        A_inv = self.A_inv[pool_idx] # pick the matrices from A_inv corresponding to the pool indices 
        b = self.b[pool_idx] # pick b vectors corresponding to the articles in the pool_idx
 
        n_pool = len(pool_idx)

        user = np.array([user] * n_pool) # construct an array which consists of user features for every item in the pool
        if self.context == 1:
            x = user    # if context is for the user then only the user features are to be used
        else:
            x = np.hstack((user, dataset.a_features[pool_idx])) # otherwise concatenate user and article features

        x = x.reshape(n_pool, self.n_features, 1)

        theta = A_inv @ b #calculates theta parameter(vector) for each article

        p = np.transpose(theta, (0, 2, 1)) @ x + self.alpha * np.sqrt(np.transpose(x, (0, 2, 1)) @ A_inv @ x) #calculate payoff matrix consisiting of payoffs of all articles
        return np.argmax(p) #get the article with maximum payoff

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters Passed:
        ---------
        displayed :displayed article index relative to the pool
        reward : user click- 0 or 1
        user :user features
        pool_idx :pool indexes for article identification
        """

        a = pool_idx[displayed]  # displayed article's index
        if self.context == 1: 
            x = np.array(user) # if context is that of user then use user's features only
        else:
            x = np.hstack((user, dataset.features[a])) #otherwise use both article and user features

        x = x.reshape((self.n_features, 1))

        self.A[a] += x @ x.T                #updating the parameters as per the chosen article
        self.b[a] += reward * x
        self.A_inv[a] = np.linalg.inv(self.A[a])


class ThompsonSampling:
    """
    Implementation of Thompson Sampling
    """

    def __init__(self):
        self.algorithm = "TS"
        self.alpha = np.ones(dataset.n_arms)
        self.beta = np.ones(dataset.n_arms)

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters Passed:
        ----------
        t :number of trial
        user :user features
        pool_idx :pool indexes for article identification
        """

        theta = np.random.beta(self.alpha[pool_idx], self.beta[pool_idx]) #here the theta vector is sampled from beta probability distribution instead of normal
        return np.argmax(theta)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters Passed:
        ----------
        displayed :displayed article index relative to the pool
        reward : user click- 0 or 1
        user :user features
        pool_idx :pool indexes for article identification
        """

        a = pool_idx[displayed] # index of the displayed article

        self.alpha[a] += reward #update parameters according to the chosen article
        self.beta[a] += 1 - reward


class Ucb1:
    """
    Implementatiom of UCB1
    """

    def __init__(self, alpha):
        """
        Parameters Passed:
        ----------
        alpha :ucb parameter
        """

        self.alpha = round(alpha, 1) #round off alpha to one decimal place
        self.algorithm = "UCB1 (α=" + str(self.alpha) + ")"

        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.ones(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters
        ----------
        t :number of trial
        user :user features
        pool_idx :pool indexes for article identification
        """

        ucbs = self.q[pool_idx] + np.sqrt(self.alpha * np.log(t + 1) / self.n[pool_idx]) #choose the arm with maximum upper confidence bound at a certian time interval
        return np.argmax(ucbs)

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b 
        Parameters passed:
        ----------
        displayed :displayed article index relative to the pool
        reward : user click 0 or 1
        user : user features
        pool_idx :pool indexes for article identification
        """

        a = pool_idx[displayed]

        self.n[a] += 1                              #update n and q acc to the chosen article
        self.q[a] += (reward - self.q[a]) / self.n[a]


class Egreedy:
    """
    Epsilon greedy algorithm implementation
    """

    def __init__(self, epsilon):
        """
        Parameters Passed:
        ----------
        epsilon :Egreedy parameter
        """

        self.e = round(epsilon, 1)  # epsilon parameter for Egreedy rounded off to one decimal place
        self.algorithm = "Egreedy (ε=" + str(self.e) + ")"
        self.q = np.zeros(dataset.n_arms)  # average reward for each arm
        self.n = np.zeros(dataset.n_arms)  # number of times each arm was chosen

    def choose_arm(self, t, user, pool_idx):
        """
        Returns the best arm's index relative to the pool
        Parameters Passed:
        ----------
        t :number of trial
        user :user features
        pool_idx :pool indexes for article identification
        """

        p = np.random.rand()   #generate a random number
        if p > self.e:         # if p>epsilon then we choose the arm with maximum average reward
            return np.argmax(self.q[pool_idx]) 
        else:
            return np.random.randint(low=0, high=len(pool_idx)) #else we choose a random arm

    def update(self, displayed, reward, user, pool_idx):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters Passed:
        ----------
        displayed :displayed article index relative to the pool
        reward : user click 0 or 1
        user :user features
        pool_idx :pool indexes for article identification
        """

        a = pool_idx[displayed] #get the chosen article's index

        self.n[a] += 1 #update the parameters
        self.q[a] += (reward - self.q[a]) / self.n[a]