
import tensorflow as tf
import tensorflow_probability as tfp

class sde(tf.keras.Model):
    """pathwise derivative implementation of nonlinear SDE using Euler-Maruyama discretization"""
    def __init__(self, dim, pastlen=1):
        """
            dim: dimension of SDE. Does not include any dimensions corresponding to periodic inputs.
                e.g. Dimension is 10, problem has periodicity in days, so there are 2 extra dimensions
                corresponding to this periodicity. We predict 10 values every timestep, then add the 2
                extra time dimensions. self.dim = 10, but there are 12 entries in each prediction.
            pastlen: number of past timesteps used to calculate drift/diffusion (1 uses current timestep only)
        """
        super().__init__()
        self.dim = dim
        self.pastlen = pastlen

        # drift architecture
        self.drift_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.drift_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.drift_dense3 = tf.keras.layers.Dense(128, activation='relu')
        self.drift_output = tf.keras.layers.Dense(dim, activation=None)
        # diffusion architecture
        self.diff_dense1 = tf.keras.layers.Dense(128, activation='relu')
        self.diff_dense2 = tf.keras.layers.Dense(128, activation='relu')
        self.diff_output = tf.keras.layers.Dense(int(dim*(dim+1)/2), activation=None)

    def obj(self, init_state, ntimesteps, yhat, start=0):
        """Calculates objective value.
        Inputs:
            init_state: list of tensors, each tensor has shape (batch size, dimension), list has shape
                (ntimesteps, batch_size, dimension). init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
            yhat: target. list of tensors, in shape of (ntimesteps, batch size, dimension)
        Returns:
            objective value calculated from self.loss
        """
        self.solve(init_state, ntimesteps, start=start)
        obj = [None for i in range(ntimesteps)]
        curobj = 0
        for i in reversed(range(ntimesteps)):
            curobj += self.loss(self.mem[i+self.pastlen], yhat[i])
            obj[i] = curobj
        return obj[-1]

    def grad(self, init_state, ntimesteps, yhat, start=0, return_obj=True):
        """Calculates objective value and gradient."""
        # this version does not use gradient tape on forward pass; this prevents storing activations in
        # memory. The trade off is the equivalent of an extra forward pass in the reverse pass.

        # forward pass
        self.solve(init_state, ntimesteps, start=start)
        obj = [None for i in range(ntimesteps)]
        curobj = 0
        for i in reversed(range(ntimesteps)):
            curobj += self.loss(self.mem[i+self.pastlen], yhat[i])
            obj[i] = curobj

        # reverse pass
        pastlen = self.pastlen
        lam = [0 for i in range(pastlen+1)]  # initialize lambda
        # lam[-1] stores the current adjoint variable; lam[:-1] accumulates terms from pastlen
        lam[-1] = 2*(self.mem[-1] - yhat[-1])
        ghat = [0 for i in range(len(self.trainable_variables))] # initialize gradient

        for i in reversed(range(ntimesteps)):
            # grab x_{i-1}, z_i
            xim1 = self.mem[i:i+pastlen]  # read as x_{i-1}
            zi = self.zmem[i]
            # calculate vector jacobian products
            with tf.GradientTape() as g:
                g.watch(xim1)
                temp = self.step(xim1, zi, tf.convert_to_tensor(start+i, dtype=tf.float32))
            vjp = g.gradient(temp, [xim1, self.trainable_variables], output_gradients=lam[-1])
            dfdx = 2*(self.mem[i+pastlen] - yhat[i])  # read as \dfrac{\partial f}{\partial x}

            # update gradient
            for j in range(len(ghat)):
                ghat[j] += vjp[1][j]
            # update adjoint variables
            if i > 0:
                lam.pop(-1)
                for j in range(pastlen):
                    lam[j] += vjp[0][j]
                lam[-1] += dfdx
                lam.insert(0, 0)

        # rescale gradient by batch size - seems tf does not do this automatically in this case
        batch_size = tf.cast(tf.shape(init_state)[1], tf.float32)
        for j in range(len(ghat)):
            ghat[j] = ghat[j] / batch_size

        if return_obj:
            return obj[-1], ghat
        else:
            return ghat

    @tf.function
    def loss(self, y, yhat):
        return tf.math.reduce_mean(tf.math.square(y - yhat))

    @tf.function
    def step(self, curstate, z, t):
        """
            curstate: shape (pastlen, batch size, dimension). current measurements + history
            z: shape (batch size, dimension, 1) where each entry is normally distributed
            t: time index of prediction
        """
        batch_size, dim_maybe_with_t = tf.shape(curstate)[1], tf.shape(curstate)[2]
        last_curstate = curstate[-1][:,:self.dim]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, dim_maybe_with_t*self.pastlen))

        drift = self.drift_dense1(curstate)
        drift = self.drift_dense2(drift)
        drift = self.drift_dense3(drift)
        drift = self.drift_output(drift)
        # tf.assert_equal(tf.shape(drift), (batch_size, self.dim))

        diff = self.diff_dense1(curstate)
        diff = self.diff_dense2(diff)
        diff = self.diff_output(diff)
        # tf.assert_equal(tf.shape(diff), (batch_size, self.dim*self.dim))

        return self.add_periodic_input_to_curstate(
            last_curstate + drift + tf.squeeze(tf.matmul(tfp.math.fill_triangular(diff),z),axis=-1), t)

    @tf.function
    def add_periodic_input_to_curstate(self, curstate, t):
        """Curstate has shape (batch size, self.dim). t is current time index. Add time to curstate."""
        return curstate

    def solve(self, init_state, ntimesteps, start=0):
        """
            init_state: length pastlen list of tensors, each tensor has shape (batch size, dimension)
                init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
            start: time index of first prediction (init_state[-1]+1). Assumed same for entire batch.
        """
        # tf.assert_equal(tf.shape(init_state)[0::2], (self.pastlen, self.dim))
        batch_size = tf.shape(init_state)[1]
        self.curstate = init_state

        self.mem = []  # list of measurements, each measurement has shape (batch size, dimension)
        self.mem.extend(init_state)
        self.zmem = []  # list of noise, each noise has shape (batch size, dimension, 1)

        for i in range(ntimesteps):
            z = tf.random.normal((batch_size, self.dim, 1))
            nextstate = self.step(self.curstate, z, tf.convert_to_tensor(start+i, dtype=tf.float32))  # call to model

            self.mem.append(nextstate)
            self.zmem.append(z)

            self.curstate = self.mem[-self.pastlen:]

