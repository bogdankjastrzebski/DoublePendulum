# DoublePendulum
Double pendulum estimation with Pyro.

## TODO

- [x] make simulation
- [x] rewrite in pytorch
- [x] calculate gradient with pytorch
    - [x] test gradient with finite difference estimator.
        Has very good results. Surprisingly good.
- [ ] derive the model mathematically.
- [ ] make a statistical model for two examples.
- [ ] make a maximum-likelihood estimator.


## Derivation of the Model

The double pendulum model is a well-known example of a chaotic deterministic model, in which small changes to the state escalate and make the simulation intractable. In various situations, we wish to have a proper monitoring of a chaotic system, for instance if we wish to control it somehow. We will here tackle the problem of keeping the virtual model, the digital twin, of a theoretical real pendulum, up to date.

Let $h_t$ be the state of the model at time $t$. We take snapshots of the model
position at distant times:
* the measurements do not capture velocity, only position;
* the measurements do not capture the position exactly,
    there is an error involved. We assume that the error is Gaussian,
    but we can use any distribution we like.
We wish to estimate the latent model parameters:
* velocities;
* exact positions;
in order to be able to simulate the double pendulum. The idea is that we can never
estimate the position and velocity well enough, due to chaotic nature of the
double pendulum, however, we can monitor it and adjust our knowledge about it.

The same idea can be applied to various situations, in which we wish to control
a chaotic system. This is a simple proof of concept that can be easily extended
to more complex models.

$$
    h_{t+\Delta} = \text{simulation}(h_t, \Delta)
$$
where $h$ is the true hidden state of the pendulum we with to estimate.
The observed variable is $x$
$$
    x_t \overset{d}{=} \text{project}(h_t) + \xi
$$
where $\xi$ is a random variable that models the imprecision of our measurement,
and the project function takes the hidden state, and selects only position of the
pendulum.



