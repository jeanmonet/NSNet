import argparse


def add_model_options(parser):
    parser.add_argument('--model', type=str, default='NSNet', help='Model choice')

    parser.add_argument('--dim', type=int, default=64, help='Dimension of variable and clause embeddings')
    parser.add_argument('--n_rounds', type=int, default=10, help='Number of rounds of message passing')
    parser.add_argument('--n_mlp_layers', type=int, default=3, help='Number of layers in all MLPs')
    parser.add_argument('--activation', type=str, default='relu', help='Activation function in all MLPs')


# --- Helper for manually passing arguments (in Jupyter) ---

class ArgOpts:
    def __init__(self, **kwargs) -> None:
        self._keys = kwargs.keys()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({', '.join(f'{key}={getattr(self, key)!r}' for key in self._keys)})"

    @classmethod
    def get_default(cls):
        return cls(
            train_dir="/opt/files/maio2022/SAT/NSNet/SATSolving/SATLIB",
            batch_size=32,
            num_workers=4,
            model="NSNet",
            task="sat-solving",

            # Page 2 (https://arxiv.org/pdf/2211.03880.pdf)
            # Existing neural SAT solvers [ 2, 3] aim to predict a single satisfying assignment for a satisfiable
            # formula. However, there can be multiple satisfying solutions, making it unclear which particular
            # solution should be generated. Instead of directly predicting a solution, NSNet performs marginal
            # inference in the solution space of a SAT problem, estimating the assignment distribution of each
            # variable among all satisfying assignments. Although NSNet is not directly trained to solve a SAT
            # problem, its estimated marginals can be used to quickly generate a satisfying assignment. One simple
            # way is to round the estimated marginals to an initial assignment and then perform the stochastic
            # local search (SLS) on it. Our experimental evaluations on the synthetic datasets with three different
            # distributions show that NSNet’s initial assignments can not only solve much more instances than
            # both BP and the neural baseline but also improve the state-of-the-art SLS solver to find a satisfying
            # assignment with fewer flips.
            # Page 5
            # To address this question, we leverage NSNet to perform marginal inference, i.e., computing the
            # marginal distribution of each variable among all satisfying assignments. In other words, instead of
            # solving a SAT problem directly, we aim to estimate the fraction of each variable that takes 1 or 0 in
            # the entire solution space. Note the marginal for each variable takes all feasible solutions into account
            # and is unique, which is more stable and interpretable to be reasoned by the neural networks. Similar
            # to Equation 3 used by BP to compute variable beliefs, NSNet estimates each marginal value bi(xi)
            # by aggregating the clause to variable assignment messages through a MLP and a softmax function: ...
            # To train NSNet to perform marginal inference accurately, we minimize the Kullback-Leibler (KL)
            # divergence loss between the estimated marginal distributions and the ground truth. We use an efficient
            # ALLSAT solver to enumerate all the satisfying assignments and take the average of them to compute
            # the true marginals.
            # Now we consider how to generate a satisfying assignment after obtaining the estimated marginals.
            # ...
            loss="marginal",       # loss type for SAT solving: "marginal" or "assignment"
            train_size=None,       # Number of training data
        )
