#Python code of Dynamic Mode Decomposition
import numpy as np
class DMD(object):
  def y_plus(self,y_data):
    """ the encoded output data without the first time series state. """
    return y_data[:, 1:]

  def y_minus(self,y_data):
    """ the encoded output data without the last time series state. """
    return y_data[:, :-1]

  def get_amat(self,y_data,r):
    """ Compute DMD A matrix by the singular value decomposition. """
    X2 = self.y_plus(y_data)
    X1 = self.y_minus(y_data)
    # compute A using the pseudoinverse.
    # return y_plus2 @ tf.linalg.pinv(y_minus2)
    ## Perform singular value decomposition on X1
    u, s, v = np.linalg.svd(X1, full_matrices=False)
    print("Singular value matrix's shape",s.shape)
    print("Singular value array",s)
    ## Compute the Koopman matrix
    A_tilde = u[:, : r].conj().T @ X2 @ v[: r, :].conj().T * np.reciprocal(s[: r])
    ## Perform eigenvalue decomposition on A_tilde
    Phi, Q = np.linalg.eig(A_tilde)
    ## Compute the coefficient matrix
    Psi = X2 @ v[: r, :].conj().T @ np.diag(np.reciprocal(s[: r])) @ Q
    A = Psi @ np.diag(Phi) @ np.linalg.pinv(Psi)
    return A

  def get_predicted_y(self,y_data,r,pred_step):
    #data=np.array(y_data)[:,0:-pred_step]
    data = np.array(y_data)[:,:]
    N, T = data.shape
    A = self.get_amat(data, r)
    print(A.shape)
    #print(A)
    mat2 = np.append(data, np.zeros((N, pred_step)), axis=1)
    #print(mat2.shape)
    for s in range(pred_step):
      mat2[:, T + s] =(A @ mat2[:, T + s - 1]).real#
    # return mat[:, - pred_step :]
    #print(mat2[:,pred_step:].shape)
    return mat2[:,pred_step:]
