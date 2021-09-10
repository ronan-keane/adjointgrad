
import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class sde(tf.keras.Model):
    """pathwise derivative implementation of nonlinear SDE using Euler-Maruyama discretization"""
    def __init__(self, dim, pastlen=1, delta=.5, l2=.01, p=1e-4):
        """
            dim: dimension of SDE. Does not include any dimensions corresponding to periodic inputs.
                e.g. Dimension is 10, problem has periodicity in days, so there are 2 extra dimensions
                corresponding to this periodicity. We predict 10 values every timestep, then add the 2
                extra time dimensions. self.dim = 10, but there are 12 entries in each prediction.
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
        # the memory requirements are further reduced because neither the drift or diffusion (covariance) are stored in memory

        # forward pass
        obj = self.solve(init_state, ntimesteps, yhat=yhat, start=start)

        # reverse pass
        pastlen = self.pastlen
        sample_weight = tf.cast(1/ntimesteps, tf.float32)
        ghat = [0 for i in range(len(self.trainable_variables))] # initialize gradient
        lam = [self.init_lambda() for i in range(pastlen+1)]  # initialize adjoint variables
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
                xi_and_extra = self.step(xim1, zi, tf.convert_to_tensor(start+i, dtype=tf.float32))
            with tf.GradientTape() as gg:
                gg.watch(xi_and_extra)
                f = self.loss(xi_and_extra, yhati, sample_weight=sample_weight)
            dfdx = gg.gradient(f, xi_and_extra)  # read as \dfrac{\partial f}{\partial x}
            lam[-1] = self.add_dfdx_to_lambda(lam[-1], dfdx)
            vjp = g.gradient(xi_and_extra, [xim1, self.trainable_variables], output_gradients=lam[-1])

            # update gradient
            for j in range(len(ghat)):
                ghat[j] += vjp[1][j]
            # update adjoint variables
            if i > 0:
                lam.pop(-1)
                for j in range(pastlen):
                    lam[j][0] += vjp[0][j]
                lam.insert(0, self.init_lambda())

        # add l2 regularization
        for j in range(len(ghat)):
            ghat[j] += self.l2*self.trainable_variables[j]

        if return_obj:
            return obj[0], ghat
        else:
            return ghat

    def init_lambda(self):
        return [0, 0]

    def add_dfdx_to_lambda(self, lam, dfdx):
        lam[0] += dfdx[0]
        lam[1] += dfdx[1]
        return lam

    @tf.function
    def loss(self, y, yhat, sample_weight=None):
        """"y should be the output of self.step. Returns loss for current prediction."""
        entropy = tf.math.reduce_mean(tf.math.log(y[1]))
        return self.loss_fn(yhat, y[0], sample_weight=sample_weight) - entropy*sample_weight*self.p

    @tf.function
    def drift(self, curstate):
        drift1 = self.drift_dense1(curstate)
        drift = self.drift_dense2(drift1)
        drift = self.drift_dense3(drift) + drift1
        drift = tf.keras.activations.relu(drift)
        return self.drift_output(drift)

    @tf.function
    def diffusion(self, curstate):
        diff1 = self.diff_dense1(curstate)
        diff = self.diff_dense2(diff1)
        diff = self.diff_dense3(diff) + diff1
        diff = tf.keras.activations.relu(diff)
        diff = self.diff_output(diff)
        diff = tfp.math.fill_triangular(diff)
        # to enforce positive definitness, need to make diagonal positive
        diag = tf.math.exp(tf.linalg.diag_part(diff))
        return tf.linalg.set_diag(diff, diag)

    @tf.function
    def step(self, curstate, z, t):
        """
        The equivalent of the h_i function.
            curstate: shape (pastlen, batch size, dimension). current measurements + history
                (curstate is equivalent to x_{i-1})
            z: shape (batch size, dimension, 1) where each entry is normally distributed
            t: (batch size,) tensor with time index of each prediction
        """
        batch_size, dim_maybe_with_t = tf.shape(curstate)[1], tf.shape(curstate)[2]
        last_curstate = curstate[-1][:,:self.dim]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, dim_maybe_with_t*self.pastlen))

        # call to model architecture
        drift = self.drift(curstate)
        diff = self.diffusion(curstate)


        diff_det = tf.math.reduce_prod(tf.linalg.diag_part(diff), axis=1)  # determinant
        out = self.add_time_input_to_curstate(
            last_curstate + drift + tf.squeeze(tf.matmul(diff,z),axis=-1), t)

        return out, diff_det  # returns tuple of (x_i, determinant) - det is used in loss

    @tf.function
    def add_time_input_to_curstate(self, curstate, t):
        """Curstate has shape (batch size, self.dim). t is current time index. Add time to curstate."""
        return curstate

    def solve(self, init_state, ntimesteps, yhat=None, start=0):
        """
        Inputs:
            init_state: list of tensors, each tensor has shape (batch size, dimension), list has shape
                (ntimesteps, batch_size, dimension). init_state are the initial conditions for SDE
            ntimesteps: integer number of timesteps
            yhat: target. list of tensors, in shape of (ntimesteps, batch size, dimension)
            start: float32 tensor with shape (batch_size,) each entry is the time index of first prediction
                for that sample
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

class sde_mle(sde):
    """Uses MLE loss instead of Huber loss + entropy maximization."""
    @tf.function
    def step(self, curstate, z, t):
        """
        The equivalent of the h_i function.
            curstate: shape (pastlen, batch size, dimension). current measurements + history
                (curstate is equivalent to x_{i-1})
            z: shape (batch size, dimension, 1) where each entry is normally distributed
            t: (batch size,) tensor with time index of each prediction
        """
        batch_size, dim_maybe_with_t = tf.shape(curstate)[1], tf.shape(curstate)[2]
        last_curstate = curstate[-1][:,:self.dim]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, dim_maybe_with_t*self.pastlen))

        # call to model architecture
        drift = self.drift(curstate)
        diff = self.diffusion(curstate)

        mu = last_curstate + drift
        out = self.add_time_input_to_curstate(
            mu + tf.squeeze(tf.matmul(diff,z),axis=-1), t)

        return out, mu, diff

    def init_lambda(self):
        if hasattr(self, 'lambda_x_init') and tf.shape(self.lambda_x_init)[0] == tf.shape(self.mem[0])[0]:
            return [self.lambda_x_init,0,0]  # step returns x_i, drift, diffusion, lambda needs same shape
        self.lambda_x_init = tf.zeros(tf.shape(self.mem[0]))
        return [self.lambda_x_init, 0, 0]

    def add_dfdx_to_lambda(self, lam, dfdx):  # loss depends on drift, diffusion only, dfdx[0] = None
        lam[1] += dfdx[1]
        lam[2] += dfdx[2]
        return lam

    @tf.function
    def loss(self, y, yhat, sample_weight=None):
        batch_size = tf.shape(yhat)[0]
        yhat = yhat[:,:self.dim]
        mu = y[1]
        sigma_chol = y[2]

        det = tf.math.reduce_prod(tf.linalg.diag_part(sigma_chol), axis=1)  # square root of determinant

        sigma_chol = tf.linalg.triangular_solve(
            sigma_chol, tf.eye(self.dim, batch_shape=[batch_size]))  # inverse of cholesky factor of covariance matrix
        mle = tf.matmul(tf.expand_dims(yhat - mu, axis=1), sigma_chol)
        mle = tf.squeeze(tf.matmul(mle, mle, transpose_b=True), [1, 2])

        return tf.reduce_mean(.5*mle+tf.math.log(det))*sample_weight


class jump_ode(tf.keras.Model):
    """ODE with discrete jumps."""
    def __init__(self, dim, jumpdim, pastlen=1, delta=.5, l2=.01):
        """
            dim: dimension of ODE. Does not include any dimensions corresponding to periodic inputs.
                e.g. Dimension is 10, problem has periodicity in days, so there are 2 extra dimensions
                corresponding to this periodicity. We predict 10 values every timestep, then add the 2
                extra time dimensions. self.dim = 10, but there are 12 entries in each prediction.
                periodic inputs are controled by add_periodic_inputs_to_curstate
            jumpdim: discrete choices for jumps. Must be odd and at least 3. At every timestep, there are
                (jumpdim-1)/2 possibilities for what the positive jump will be, likewise for negative jumps,
                and the last possibility is no jump.
            pastlen: number of past timesteps used to calculate next step (1 uses current timestep only)
            delta: for Huber loss https://www.tensorflow.org/api_docs/python/tf/keras/losses/Huber
            l2: strength of l2 regularization (l2*self.trainable_variables is added to gradient)
        """
        super().__init__()
        assert type(dim) == int
        assert type(jumpdim) == int
        assert type(pastlen) == int
        assert dim>0
        assert jumpdim > 1
        assert jumpdim % 2 == 1
        assert pastlen >0
        self.dim = dim
        self.jumpdim = jumpdim
        self.pastlen = pastlen
        self.njumps = (self.jumpdim-1)//2  # number of possibilities for each of positive, negative jumps

        # drift architecture
        self.drift_dense1 = tf.keras.layers.Dense(100, activation='tanh')
        self.drift_dense2 = tf.keras.layers.Dense(100, activation='tanh')
        self.drift_dense3 = tf.keras.layers.Dense(100, activation=None)
        self.drift_output = tf.keras.layers.Dense(dim, activation=None)
        # jumps architecture
        self.jumps_dense1 = tf.keras.layers.Dense(100, activation='tanh')
        self.jumps_dense2 = tf.keras.layers.Dense(100, activation='tanh')
        self.jumps_dense3 = tf.keras.layers.Dense(100, activation=None)
        self.jumps_output = tf.keras.layers.Dense(dim*(2*jumpdim-1), activation=None)

        self.loss_fn = tf.keras.losses.Huber(delta=delta)
        self.loss_no_reduction = tf.keras.losses.Huber(delta=delta, reduction=tf.keras.losses.Reduction.NONE)
        self.l2 = tf.cast(l2, tf.float32)

        # simple baseline
        self.baseline = SimpleBaseline()

    @tf.function
    def drift(self, curstate):
        drift1 = self.drift_dense1(curstate)
        drift = self.drift_dense2(drift1)
        drift = self.drift_dense3(drift) + drift1
        drift = tf.keras.activations.relu(drift)
        return self.drift_output(drift)

    @tf.function
    def jumps(self, curstate):
        jump1 = self.jumps_dense1(curstate)
        jumps = self.jumps_dense2(jump1)
        jumps = self.jumps_dense3(jumps) + jump1
        jumps = tf.keras.activations.relu(jumps)
        jumps = self.jumps_output(jumps)  # output logits and jump magnitudes

        logits = jumps[:, :self.dim*self.jumpdim]
        posjumps = tf.math.exp(jumps[:, self.dim*self.jumpdim:self.dim*(self.jumpdim+self.njumps)])
        negjumps = -tf.math.exp(jumps[:, self.dim*(self.jumpdim+self.njumps):])
        return logits, posjumps, negjumps

    @tf.function
    def add_time_input_to_curstate(self, curstate, t):
        """Curstate has shape (batch size, self.dim). t is tensor of current time index. Add time to curstate."""
        return curstate

    @tf.function
    def loss(self, pred, true, sample_weight=None):
        """Returns loss for the current prediction."""
        return self.loss_fn(true, pred, sample_weight=sample_weight)

    @tf.function
    def loss_over_batch(self, pred, true, sample_weight=None):  # Does not average over batch.
        return self.loss_no_reduction(true, pred, sample_weight=sample_weight)

    @tf.function
    def step(self, curstate, t, use_y=None):
        """
        The equivalent of both the h_i and p_i(y_i) function.
            curstate: shape (pastlen, batch size, dimension). current measurements + history
                (curstate is equivalent to x_{i-1})
            t: (batch size,) tensor with time index of each prediction
            use_y: if None, we sample y according to p_i(y_i), and return x_i, entropy, y. If use_y=y,
                then we return x_i, entropy, \log p_i(y).
                use_y=None is used in the forward pass and we keep y in memory. Reverse pass uses
                use_y=y so that we can calculate the relevant vjp.
        """
        batch_size, dim_maybe_with_t = tf.shape(curstate)[1], tf.shape(curstate)[2]
        last_curstate = curstate[-1][:,:self.dim]  # most recent measurements
        curstate = tf.transpose(curstate, [1,0,2])
        curstate = tf.reshape(curstate, (batch_size, dim_maybe_with_t*self.pastlen))

        # call to model
        drift = self.drift(curstate)
        logits, posjumps, negjumps = self.jumps(curstate)
        # reshaping logits/jumps
        logits = tf.reshape(logits, (batch_size*self.dim, self.jumpdim))
        posjumps = tf.reshape(posjumps, (batch_size, self.dim, self.njumps))
        negjumps = tf.reshape(negjumps, (batch_size, self.dim, self.njumps))
        jumps = tf.concat([self.jumps_zero_pad, posjumps, negjumps], axis=2)

        # sample y, which represents which jump category we take
        if use_y==None:
            y = tf.random.categorical(logits, 1, dtype=tf.int32)
        else:
            y = use_y
        # select corresponding jumps from y
        inds = tf.concat([self.jumps_ind_pad, y], axis=1)
        jumps = tf.gather_nd(jumps, inds)
        jumps = tf.reshape(jumps, (batch_size, self.dim))

        # the next state
        out = self.add_time_input_to_curstate(
            last_curstate + drift + jumps, t)

        if use_y==None: # forward pass
            return out, y
        else:  # reverse
            # create log probability of the jumps corresponding to y
            probs = tf.exp(logits)
            probs = probs/tf.reshape(tf.reduce_sum(probs, axis=1), (batch_size*self.dim, 1))  # normmalize
            probs = tf.reshape(probs, (batch_size, self.dim, self.jumpdim))
            probs = tf.math.log(probs)
            probs = tf.gather_nd(probs, inds)
            probs = tf.reshape(probs, (batch_size, self.dim))
            probs = tf.reduce_sum(probs, axis=1)
            return out, probs

    def solve(self, init_state, ntimesteps, true=None, start=0, loss_output='scalar'):
        """
        Inputs:
            init_state: list of tensors, each tensor has shape (batch size, dimension), list has shape
                (ntimesteps, batch_size, dimension). init_state are the initial conditions for process.
            ntimesteps: integer number of timesteps
            true: target. list of tensors, in shape of (ntimesteps, batch size, dimension)
            start: float32 tensor with shape (batch_size,) each entry is the time index of first prediction
                for that sample
            loss_output: if 'scalar' returns mean loss value over batch. If 'batch', returns loss value for
                each sample in batch.
        Returns:
            cumulative objective values (when true is not None). Shape is (ntimesteps,) if loss_output='scalar'
            otherwise shape is (batch_size, ntimesteps)
        """
        batch_size = tf.shape(init_state)[1]
        self.curstate = init_state

        self.mem = []  # list of measurements, each measurement has shape (batch size, dimension)
        self.mem.extend(init_state)
        self.ymem = []  # list of jump choices, each choice has shape (batch size*dimension,1) and corresponds
        # to the index of the selected jump
        self.jumps_zero_pad = tf.zeros((batch_size, self.dim, 1))
        self.jumps_ind_pad = tf.stack([tf.repeat(tf.range(batch_size), [self.dim]),
                         tf.tile(tf.range(self.dim), [batch_size])], axis=1)

        obj = [] if true is not None else None
        loss = self.loss if loss_output=='scalar' else self.loss_over_batch
        sample_weight = tf.cast(1/ntimesteps, tf.float32)

        for i in range(ntimesteps):
            nextstate, y = self.step(
                self.curstate, tf.convert_to_tensor(start+i, dtype=tf.float32))
            if true is not None:
                obj.append(loss(nextstate, true[i], sample_weight=sample_weight))

            self.mem.append(nextstate)
            self.ymem.append(y)

            self.curstate = self.mem[-self.pastlen:]

        if true is not None:
            if loss_output !='scalar':
                obj = tf.stack(obj, axis=1)
            obj = tf.math.cumsum(obj, reverse=True, axis=-1)
        return obj

    def grad(self, init_state, ntimesteps, true, start=0, return_obj=True):
        """Calculates objective value and gradient."""

        # forward pass
        obj = self.solve(init_state, ntimesteps, true=true, start=start, loss_output='batch')

        self.baseline.update(obj)

        # reverse pass
        pastlen = self.pastlen
        sample_weight = tf.cast(1/ntimesteps, tf.float32)
        ghat = [0 for i in range(len(self.trainable_variables))] # initialize gradient
        lam = [self.init_lambda(i, ntimesteps, obj) for i in range(ntimesteps-1-pastlen,ntimesteps)]
        # lam[-1] stores the current adjoint variable; lam[:-1] accumulates terms from pastlen
        # Note that lambda_i needs to have the same shape as step when use_y = y

        for i in reversed(range(ntimesteps)):
            xim1 = self.mem[i:i+pastlen]
            yi = self.ymem[i]
            truei = true[i]

            with tf.GradientTape() as g:
                g.watch(xim1)
                xi_and_extra =  self.step(xim1, tf.convert_to_tensor(start+i, dtype=tf.float32), use_y=yi)
            xi = xi_and_extra[0]
            with tf.GradientTape() as gg:
                gg.watch(xi)
                f = self.loss(xi, truei, sample_weight=sample_weight)
            dfdx = gg.gradient(f, xi)
            lam[-1][0] += dfdx[0]
            vjp = g.gradient(xi_and_extra, [xim1, self.trainable_variables], output_gradients=lam[-1])

            # update gradient
            for j in range(len(ghat)):
                ghat[j] += vjp[1][j]
            # update adjoint variables
            if i > 0:
                lam.pop(-1)
                for j in range(pastlen):
                    lam[j][0] += vjp[0][j]
                lam.insert(0, self.init_lambda(i-1-pastlen, ntimesteps, obj))

        # add l2 regularization
        for j in range(len(ghat)):
            ghat[j] += self.l2*self.trainable_variables[j]

        if return_obj:
            return tf.reduce_mean(obj[:,0]), ghat
        else:
            return ghat

    def init_lambda(self, ind, ntimesteps, obj):
        """Initialize lambda_{ind}."""
        ind = max(ind, 0)
        batch_size = tf.shape(obj)[0]
        baselines = tf.repeat(self.baseline(ind, ntimesteps), batch_size)
        return [0, obj[:,ind] - baselines]


class SimpleBaseline:
    """Simple Baseline with no parameters.

    Assumes that loss is summable, i.e. objective = \sum_{i=1}^n f_i(x_i)
    """
    def __init__(self, alpha=.02):
        """alpha = decay of exponential moving average (higher = faster decay)"""
        self.baseline = np.zeros((0,), dtype=np.single)
        self.alpha = alpha

    def __call__(self, ind, ntimesteps):
        """
            ind: index of of current requested baseline (must be less than ntimesteps)
            ntimesteps: number of total timesteps in current forward solve
        """
        return self.baseline[ind-ntimesteps]

    def update(self, obj):
        """Update constant baseline with new objective values.
        obj should be a tensor with shape (batch size, ntimesteps)
        """
        obj = tf.reduce_mean(obj,axis=0).numpy()
        diff = len(obj) - len(self.baseline)
        if diff > 0:
            if len(self.baseline) > 0:
                self.baseline = self.alpha*obj[diff:] + (1-self.alpha)*self.baseline
            self.baseline = np.concatenate((obj[:diff], self.baseline), axis=0)
        elif diff < 0:
            self.baseline[diff:] = self.alpha*obj + (1-self.alpha)*self.baseline[diff:]
        else:
            self.baseline = self.alpha*obj + (1-self.alpha)*self.baseline

class NoBaseline:
    def __init__(self):
        pass
    def __call__(self, *args):
        return 0
    def update(self, *args):
        pass

