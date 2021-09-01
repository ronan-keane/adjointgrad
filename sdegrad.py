
import tensorflow as tf
import tensorflow_probability as tfp

class sde(tf.keras.Model):
    """pathwise derivative implementation of nonlinear SDE using Euler-Maruyama discretization"""
    def __init__(self, dim, pastlen=1, delta=.5, l2=.01, p=1e-4):
        """
            dim: dimension of SDE. Does not include any dimensions corresponding to periodic inputs.
                e.g. Dimension is 10, problem has periodicity in days, so there are 2 extra dimensions
                corresponding to this periodicity. We predict 10 values every timestep, then add the 2
                extra time dimensions. self.dim = 10, but there are 12 entries in each prediction.
                (so the dimension is treated as 12.)
            pastlen: number of past timesteps used to calculate drift/diffusion (1 uses current timestep only)
            delta: for Huber loss https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber
            l2: strength of l2 regularization (l2*self.trainable_variables is added to gradient)
            p: weight of entropy maximization term.
        """
        super().__init__()
        self.dim = dim
        self.pastlen = pastlen

        # drift architecture
        self.drift_dense1 = tf.keras.layers.Dense(100, activation='tanh')
        self.drift_dense2 = tf.keras.layers.Dense(100, activation='tanh')
        self.drift_dense3 = tf.keras.layers.Dense(100, activation=None)
        self.drift_output = tf.keras.layers.Dense(dim, activation=None)
        # diffusion architecture
        self.diff_dense1 = tf.keras.layers.Dense(100, activation='tanh')
        self.diff_dense2 = tf.keras.layers.Dense(100, activation='tanh')
        self.diff_dense3 = tf.keras.layers.Dense(100, activation=None)
        self.diff_output = tf.keras.layers.Dense(int(dim*(dim+1)/2), activation=None)

        self.loss_fn = tf.keras.losses.Huber(delta=delta)
        self.l2 = tf.cast(l2, tf.float32)
        self.p = tf.cast(p, tf.float32)

    def grad(self, init_state, ntimesteps, yhat, start=0, return_obj=True):
        """Calculates objective value and gradient."""
        # this version does not use gradient tape on forward pass; this prevents storing activations in
        # memory. The trade off is the equivalent of an extra forward pass in the reverse pass.

        # forward pass
        obj = self.solve(init_state, ntimesteps, yhat=yhat, start=start)

        # reverse pass
        pastlen = self.pastlen
        sample_weight = tf.cast(1/ntimesteps, tf.float32)
        ghat = [0 for i in range(len(self.trainable_variables))] # initialize gradient
        lam = [[0, 0] for i in range(pastlen+1)]  # initialize lambda
        # lam[-1] stores the current adjoint variable; lam[:-1] accumulates terms from pastlen
        # Note that lambda_i needs to have the same shape as h_i.

        for i in reversed(range(ntimesteps)):
            # grab x_{i-1}, z_i
            xim1 = self.mem[i:i+pastlen]  # read as x_{i-1}
            zi = self.zmem[i]
            yhati = yhat[i]
            # calculate vector jacobian products
            with tf.GradientTape() as g:
                g.watch(xim1)
                xi_and_det = self.step(xim1, zi, tf.convert_to_tensor(start+i, dtype=tf.float32))
            with tf.GradientTape() as gg:
                gg.watch(xi_and_det)
                f = self.loss(xi_and_det, yhati, sample_weight=sample_weight)
            dfdx = gg.gradient(f, xi_and_det)  # read as \dfrac{\partial f}{\partial x}
            lam[-1][0] += dfdx[0]
            lam[-1][1] += dfdx[1]
            vjp = g.gradient(xi_and_det, [xim1, self.trainable_variables], output_gradients=lam[-1])

            # update gradient
            for j in range(len(ghat)):
                ghat[j] += vjp[1][j]
            # update adjoint variables
            if i > 0:
                lam.pop(-1)
                for j in range(pastlen):
                    lam[j][0] += vjp[0][j]
                lam.insert(0, [0,0])

        # # rescale gradient by batch size
        # batch_size = tf.cast(tf.shape(init_state)[1], tf.float32)
        # for j in range(len(ghat)):
        #     ghat[j] = ghat[j] / batch_size

        # add l2 regularization
        for j in range(len(ghat)):
            ghat[j] += self.l2*self.trainable_variables[j]

        if return_obj:
            return obj[0], ghat
        else:
            return ghat

    @tf.function
    def loss(self, y, yhat, sample_weight=None):
        """"y should be the output of self.step. Returns loss for current prediction."""
        entropy = tf.math.reduce_mean(tf.math.log(y[1]))
        return self.loss_fn(yhat, y[0], sample_weight=sample_weight) - entropy*sample_weight*self.p

    @tf.function
    def step(self, curstate, z, t):
        """
        The equivalent of the h_i function.
            curstate: shape (pastlen, batch size, dimension). current measurements + history
                (curstate is equivalent to x_{i-1})
            z: shape (batch size, dimension, 1) where each entry is normally distributed
            t: time index of prediction
        """
        batch_size, dim_maybe_with_t = tf.shape(curstate)[1], tf.shape(curstate)[2]
        last_curstate = curstate[-1][:,:self.dim]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, dim_maybe_with_t*self.pastlen))
        # drift
        drift1 = self.drift_dense1(curstate)
        drift = self.drift_dense2(drift1)
        drift = self.drift_dense3(drift + drift1)
        drift = tf.keras.activations.relu(drift)
        drift = self.drift_output(drift)
        # diffusion
        diff1 = self.diff_dense1(curstate)
        diff = self.diff_dense2(diff1)
        diff = self.diff_dense3(diff + diff1)
        diff = tf.keras.activations.relu(diff)
        diff = self.diff_output(diff)
        diff = tfp.math.fill_triangular(diff)
        # to enforce positive definitness, need to make diagonal positive
        diag = tf.math.exp(tf.linalg.diag_part(diff))
        diff = tf.linalg.set_diag(diff, diag)
        diff_det = tf.math.reduce_prod(diag, axis=1)  # determinant
        # diff_det = tf.ones((batch_size, self.dim))

        out = self.add_periodic_input_to_curstate(
            last_curstate + drift + tf.squeeze(tf.matmul(diff,z),axis=-1), t)

        return out, diff_det  # returns tuple of (x_i, determinant) - det is used in loss

    @tf.function
    def add_periodic_input_to_curstate(self, curstate, t):
        """Curstate has shape (batch size, self.dim). t is current time index. Add time to curstate."""
        return curstate

    def solve(self, init_state, ntimesteps, yhat=None, start=0):
        """
        Inputs:
            init_state: list of tensors, each tensor has shape (batch size, dimension), list has shape
                (ntimesteps, batch_size, dimension). init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
            yhat: target. list of tensors, in shape of (ntimesteps, batch size, dimension)
            start: time index of first prediction (init_state[-1]+1). Assumed same for entire batch.
        Returns:
            objective value calculated from self.loss (when yhat is not None)
        """
        # tf.assert_equal(tf.shape(init_state)[0::2], (self.pastlen, self.dim))
        batch_size = tf.shape(init_state)[1]
        self.curstate = init_state

        self.mem = []  # list of measurements, each measurement has shape (batch size, dimension)
        self.mem.extend(init_state)
        self.zmem = []  # list of noise, each noise has shape (batch size, dimension, 1)

        obj = [] if yhat is not None else None
        sample_weight = tf.cast(1/ntimesteps, tf.float32)

        for i in range(ntimesteps):
            z = tf.random.normal((batch_size, self.dim, 1))
            nextstate = self.step(self.curstate, z, tf.convert_to_tensor(start+i, dtype=tf.float32))  # call to model
            if yhat is not None:
                obj.append(self.loss(nextstate, yhat[i], sample_weight=sample_weight))  # call to loss if requested

            self.mem.append(nextstate[0])
            self.zmem.append(z)

            self.curstate = self.mem[-self.pastlen:]

        if yhat is not None:
            obj = tf.math.cumsum(obj, reverse=True)
        return obj

