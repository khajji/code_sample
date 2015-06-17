import numpy as np

"""This class implements the EM algorithm for the mixture of multinomials. The
input is a set of documents. Each documents formed of words. """
class Mix_multi():
  """
  P:  Array [nb_topics, nb_words] where each row is topic
  Pi: Array [nb_docs, nb_topics] that associates to each document the mix coeffs
      over topics
  rsp:  dictionary such as rsp[d][w]: array [nb_topics, 1] with the rsp of each
        topic in generating word w in document d
  N:    Array [nb_docs, nb_topics] that associate to each topic and each document
        the estimation of number of realisations sampled from it
  data: dictionary data[doc_id][word_id]: nb of times word appears in doc
  """
  def __init__(self, data, nb_topics, nb_words):
    self.nb_docs = len(data)
    self.nb_topics = nb_topics # nb of topics to find
    self.nb_words = nb_words # Vocabulary size
    self.P = np.zeros((self.nb_topics, self.nb_words))
    self.Pi = np.zeros((self.nb_docs, self.nb_topics))
    self.rsp = dict()
    self.data = data

  def initialize(self):
    for d in self.data:
      self.rsp[d] = dict()
      for w in self.data[d]:
        self.rsp[d][w] = 0
    self.P = normalize_mat(np.random.rand(self.nb_topics, self.nb_words))
    self.Pi = normalize_mat(np.random.rand(self.nb_docs, self.nb_topics))

  def e_phase(self):
    """E phase: evaluate the responsabilities using the current values of
    Pi and P"""
    for d in self.rsp:
      for w in self.rsp[d]:
        p = self.Pi[d,:]*self.P[:,w] # p[k] = p[z=k|w]
        self.rsp[d][w] = p/np.sum(p)

  def m_phase(self):
    """Re-estimate the parameters Pi and P using the current responsabilities"""
    P = np.zeros(self.P.shape)
    # Compute the Expected number of words N_k per topic
    N = np.zeros((self.nb_docs, self.nb_topics))
    for d in self.rsp:
      for w in self.rsp[d]:
        N[d,:] += (self.data[d][w]*self.rsp[d][w]).transpose()
        P[:,w] += self.data[d][w]*self.rsp[d][w]

    self.P = normalize_mat(P) # Update the topics
    self.Pi = normalize_mat(N) # Update the mixture coefficients

  def likelihood(self):
    """Returns the log likelihood of the data given the model parameters"""
    ll = 0
    for d in self.data:
      for w in self.data[d]:
        ll += self.data[d][w]*np.log10(np.dot(self.Pi[d,:],self.P[:,w]))
    return ll

  def run(self, max_iter, delta):
    """Run the EM algo for the mixture of multinomials. The algo converges as
    as soon as we reach the maximum number of iterations [max_iter] or if the
    change in likleihood is less than a given threshold [delta]"""

    convergence = False
    print "MM with {0} documents and {1} topics".format(
          self.nb_docs, self.nb_topics)
    print "Running EM, Max number of iterations: {0}".format(max_iter)
    ll = [self.likelihood()]
    while not convergence:
      print "Iter {0}, likelihood {1}".format(len(ll)-1, ll[-1])
      self.e_phase()
      self.m_phase()
      ll.append(self.likelihood())
      convergence = (len(ll) > max_iter) or (ll[-1] - ll[-2] < delta)
    print "Iter {0}, likelihood {1}".format(len(ll)-1, ll[-1])


  def normalize_mat(M):
      """
      This function normalizes the matrix M so the sum of its elements along
      each row is 1

      Arguments:
      - `M`: numpy array
      """
      if len(M.shape) ==2:
        M = 1.0*np.array(M)
        s =  np.sum(M,axis=1)
        index = np.nonzero(s)[0]
        s = s[index]
        s = s.reshape(s.size,1,order='F')
        M[index,:] = M[index,:] / np.tile(s,(1,M.shape[1]))
        return M
      else:
        if len(M.shape)==1:
          return M/np.sum(M)

#for testing..
if __name__ == "__main__":
  data = dict()
  # Documents
  data[0] = {0:100, 1:29}
  data[1] = {2:100, 4:150}
  data[2] = {0:100, 1:120, 2:230, 4:10}
  k = 2 # nb of topics
  v = 5 # vocabulary size
  max_iter = 30
  delta = 0.001
  mm = Mix_multi(data, k, v)
  mm.initialize()
  mm.run(max_iter, delta)




