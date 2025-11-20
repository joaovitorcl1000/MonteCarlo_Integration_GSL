# MonteCarlo_Integration_GSL
Parallel Monte Carlo integration implementation using the **GSL (GNU Scientific Library)** and **OpenMP** for accelerated performance. This is ideal for multidimensional integrals where traditional methods are inefficient.

For illustration, the code calculates the following integral within the unit cube $[0, 1]^3$:

$$
\int_0^1 \int_0^1 \int_0^1 \left[ p(x + y + z) + q(x^2 + y^2 + z^2) \right] dx dy dz
$$

For the default constants $p = 0.1$ and $q = 0.1$, the exact result is $0.25$ (or $\frac{1}{4}$).
