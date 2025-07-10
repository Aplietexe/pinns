# PINNs

Adapted the code to PyTorch. Training time went from 2:20 minutes to 8 seconds.

![coefficient error](./hist/init_coeffs.png)
![function error](./hist/init_error.png)
![comparison](./hist/init_comp.png)

I'm thinking that learning the values of the derivatives (Maclaurin coefficients) makes it hard for gradient flow to properly reach the parameters of the network that affect higher order coefficients the most, as we are dividing those by factorial terms in the loss.
