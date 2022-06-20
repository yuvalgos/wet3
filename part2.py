import numpy as np
import matplotlib.pyplot as plt
from radial_basis_function_extractor import RadialBasisFunctionExtractor



if __name__ == "__main__":
    # q2.2
    extractor = RadialBasisFunctionExtractor([12, 10])

    x = np.linspace(-2.5, -1, 100)
    v = np.linspace(-2.5, 0, 100)
    x, v = np.meshgrid(x, v)

    features = extractor.encode_states_with_radial_basis_functions(np.array([x.reshape(-1), v.reshape(-1)]).T)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, v, features[:, 0].reshape(x.shape))
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    plt.show()

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    ax.plot_surface(x, v, features[:, 1].reshape(x.shape))
    ax.set_xlabel("x")
    ax.set_ylabel("v")
    plt.show()
    pass
