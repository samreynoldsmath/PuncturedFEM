from .poly.poly import Polynomial
from .trace import DirichletTrace
from ..mesh.cell import MeshCell


class LocalPolynomial:
    exact_form: Polynomial
    trace: DirichletTrace
    grad1: Polynomial
    grad2: Polynomial
    antilap: Polynomial
    antilap_trace: DirichletTrace

    def __init__(self, exact_form: Polynomial, K: MeshCell) -> None:
        self.exact_form = exact_form
        self.trace = DirichletTrace(
            edges=K.get_edges(), values=exact_form(*K.get_boundary_points())
        )
        self.grad1, self.grad2 = exact_form.grad()
        self.trace.set_weighted_normal_derivative(
            exact_form.get_weighted_normal_derivative(K)
        )
        self.antilap = exact_form.anti_laplacian()
        self.antilap_trace = DirichletTrace(
            edges=K.get_edges(), values=self.antilap(*K.get_boundary_points())
        )
        self.antilap_trace.set_weighted_normal_derivative(
            self.antilap.get_weighted_normal_derivative(K)
        )
