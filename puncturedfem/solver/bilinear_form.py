from ..locfun.locfun import locfun
from ..locfun.poly.piecewise_poly import piecewise_polynomial
from ..locfun.poly.poly import polynomial


class bilinear_form:
    diffusion_constant: float
    reaction_constant: float
    rhs_poly: polynomial

    def __init__(
        self,
        diffusion_constant: float,
        reaction_constant: float,
        rhs_poly: polynomial,
    ) -> None:
        self.set_diffusion_constant(diffusion_constant)
        self.set_reaction_constant(reaction_constant)
        self.set_rhs_poly(rhs_poly)

    def __str__(self) -> str:
        s = "bilinear_form:"
        s += "\n\tdiffusion_constant: %s" % self.diffusion_constant
        s += "\n\treaction_constant: %s" % self.reaction_constant
        s += "\n\trhs_poly: %s" % self.rhs_poly
        return s

    def set_diffusion_constant(self, diffusion_constant: float) -> None:
        self.diffusion_constant = diffusion_constant

    def set_reaction_constant(self, reaction_constant: float) -> None:
        self.reaction_constant = reaction_constant

    def set_rhs_poly(self, f_poly: polynomial) -> None:
        self.rhs_poly = f_poly

    def eval(self, u: locfun, v: locfun) -> float:
        h1 = u.get_h1_semi_inner_prod(v)
        l2 = u.get_l2_inner_prod(v)
        return self.diffusion_constant * h1 + self.reaction_constant * l2

    def eval_rhs(self, v: locfun) -> float:
        m = len(v.poly_trace.polys)
        poly_trace = piecewise_polynomial(
            num_polys=m,
            polys=[self.rhs_poly for _ in range(len(v.poly_trace.polys))],
        )
        f = locfun(
            v.solver, lap_poly=self.rhs_poly.laplacian(), poly_trace=poly_trace
        )
        f.compute_all()
        return f.get_l2_inner_prod(v)
