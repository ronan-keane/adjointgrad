
import tensorflow as tf

class sde(tf.keras.Model):
    """pathwise derivative implementation of nonlinear SDE using Euler-Maruyama discretization"""
    def __init__(self, dim, pastlen=1):
        """
            dim: dimension of SDE
            pastlen: number of past timesteps used to calculate drift/diffusion (1 uses current timestep only)
        """
        super().__init__()
        self.dim = dim
        self.pastlen = pastlen

        # drift architecture
        self.drift_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.drift_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.drift_output = tf.keras.layers.Dense(dim, activation=None)
        # diffusion architecture
        self.diff_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.diff_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.diff_output = tf.keras.layers.Dense(dim*dim)

    def obj(self, init_state, ntimesteps, yhat):
        """Calculates objective value.
        Inputs:
            init_state: list of tensors, each tensor has shape (batch size, dimension). pastlen tensors total.
                init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
            yhat: target. list of tensors, in shape of (ntimesteps, batch size, dimension)
        Returns:
            objective value calculated from self.loss
        """
        self.solve(init_state, ntimesteps)
        obj = []
        curobj = 0
        for i in range(ntimesteps):
            curobj += self.loss(self.mem[i+self.pastlen], yhat[i])
            obj.append(curobj)
        return obj[-1]

    def grad(self, init_state, ntimesteps, yhat, return_obj=True):
        """Calculates objective value and gradient."""
        # this version does not use gradient tape on forward pass; this prevents storing activations in
        # memory. The trade off is the equivalent of an extra forward pass in the reverse pass.

        # batch_size = tf.shape(init_state)[1]

        # forward pass
        self.solve(init_state, ntimesteps)
        obj = []
        curobj = 0
        for i in range(ntimesteps):
            curobj += self.loss(self.mem[i+self.pastlen], yhat[i])
            obj.append(curobj)

        # reverse pass
        pastlen = self.pastlen
        lam = [0 for i in range(pastlen+1)]  # initialize lambda
        # lam[-1] stores the current adjoint variable; lam[:-1] accumulates terms from pastlen
        lam[-1] = 2*(self.mem[-1] - yhat[-1])
        ghat = [0 for i in range(len(self.trainable_variables))] # gradient
        for i in reversed(range(ntimesteps)):
            # grab x_{i-1}, z_i
            xim1 = self.mem[i:i+pastlen]  # read as x_{i-1}
            zi = self.zmem[i]
            # calculate vector jacobian products
            with tf.GradientTape() as g:
                g.watch(xim1)
                temp = self.step(xim1, zi)
            vjp = g.gradient(temp, [xim1, self.trainable_variables], output_gradients=lam[-1])
            dfdx = 2*(self.mem[i+pastlen] - yhat[i])  # read as \dfrac{\partial f}{\partial x}

            # update gradient
            for j in range(len(ghat)):
                ghat[j] = ghat[j] + vjp[1][j]
            # update adjoint variables
            if i > 0:
                lam.pop(-1)
                for j in range(pastlen):
                    lam[j] += vjp[0][j]
                lam[-1] += dfdx
                lam.insert(0, 0)

        if return_obj:
            return obj[-1], ghat
        else:
            return ghat

    @tf.function
    def loss(self, y, yhat):
        return tf.math.reduce_mean(tf.math.square(y - yhat))

    @tf.function
    def step(self, curstate, z):
        """
            curstate: shape (pastlen, batch size, dimension) of the current measurements + history
            z: shape (batch size, dimension, 1) where each entry is normally distributed
        """
        batch_size = tf.shape(curstate)[1]
        last_curstate = curstate[-1]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, self.dim*self.pastlen))

        drift = self.drift_dense1(curstate)
        drift = self.drift_dense2(drift)
        drift = self.drift_output(drift)
        # tf.assert_equal(tf.shape(drift), (batch_size, self.dim))

        diff = self.diff_dense1(curstate)
        diff = self.diff_dense2(diff)
        diff = self.diff_output(diff)
        # tf.assert_equal(tf.shape(diff), (batch_size, self.dim*self.dim))

        return last_curstate + drift + tf.squeeze(
            tf.matmul(tf.reshape(diff, (batch_size, self.dim, self.dim)),z),axis=-1)

    def solve(self, init_state, ntimesteps):
        """
            init_state: length pastlen list of tensors, each tensor has shape (batch size, dimension)
                init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
        """
        tf.assert_equal(tf.shape(init_state)[0::2], (self.pastlen, self.dim))
        batch_size = tf.shape(init_state)[1]
        self.curstate = init_state

        self.mem = []  # list of measurements, each measurement has shape (batch size, dimension)
        self.mem.extend(init_state)
        self.zmem = []  # list of noise, each noise has shape (batch size, dimension, 1)

        for i in range(ntimesteps):
            z = tf.random.normal((batch_size, self.dim, 1))
            nextstate = self.step(self.curstate, z)  # call to model

            self.mem.append(nextstate)
            self.zmem.append(z)

            self.curstate = self.mem[-self.pastlen:]



test = sde(10, pastlen=3)

init_state = [tf.random.normal((1, test.dim)) for i in range(test.pastlen)]

test.solve(init_state, 30)

yhat = [tf.ones((1, test.dim)) for i in range(30)]

out = test.grad(init_state, 30, yhat)
#%%
z = tf.random.normal((test.dim,1))
a = test.step(init_state, z)






