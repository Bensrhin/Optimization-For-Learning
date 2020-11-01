"""
    quad(x,Q,q)

Compute the quadratic

	f(x) = 1/2 x'Qx + q'x

"""
function quad(x,Q,q)
	return 1/2 * x'*Q*x + q'*x
end



"""
    guadconj(y,Q,q)

Compute the convex conjugate of the quadratic

Solution :

	f*(y) = 1/2 (y - q)' Q^{-1} (y - q)

"""
function quadconj(y,Q,q)
	return 1/2 * (y - q)'*inv(Q)*(y - q)
end



"""
    box(x,a,b)

Compute the indicator function of for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.
"""
function box(x,a,b)
	return all(a .<= x .<= b) ? 0.0 : Inf
end



"""
    boxconj(y,a,b)

Compute the convex conjugate of the indicator function of for the box contraint
	a <= x <= b
where the inequalites are applied element-wise.

Solution :

	g*(y) = ∑ max_i (y_i.a_i, y_i.b_i)
"""
function boxconj(y,a,b)
	return sum(max.(y .* a, y .* b))
end



"""
    grad_quad(x,Q,q)

Compute the gradient of the quadratic

	1/2 x'Qx + q'x

Solution :

		∇ f(x) = Qx + q
"""
function grad_quad(x,Q,q)
	return Q*x + q
end



"""
    grad_quadconj(y,Q,q)

Compute the gradient of the convex conjugate of the quadratic

	1/2 x'Qx + q'x

Solution :
	∇ f*(y) = Q^{-1} (y - q)
"""
function grad_quadconj(y,Q,q)
	return inv(Q) * (y - q)
end


"""
	prox_i(xi, ai, bi) is the proximal operator of :
		the indicator function for [ai, bi]
			ai <= xi <= bi

						ai  if 	xi <= ai
prox_i(xi, ai, bi) = {  	xi  if 	ai <= xi <= bi
						bi  if 	xi >= bi

"""
function prox_i(xi, ai, bi)
	if xi <= ai
		return ai
	elseif xi >= bi
		return bi
	else
		return xi
	end
end
"""
    prox_box(x,a,b) = (prox_i(xi, ai, bi))' i ∈ {1,..,n}

Compute the proximal operator of the indicator function for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.


"""

function prox_box(x,a,b,gamma)
	return prox_i.(x, a, b)
end


"""
	prox_i_conj(xi, ai, bi, γ) is the proximal operator of
	the convex conjugate the indicator function for [ai, bi]
			ai <= xi <= bi

								xi - γ.ai    if  	xi <= γ.ai
prox_i_conj(xi, ai, bi, γ) = {   0 			 if 	γ.ai <= xi <= γ.bi
								xi - γ.bi  	 if 	xi >= γ.bi

"""
function prox_i_conj(xi, ai, bi, γ)
	if xi <= γ * ai
		return xi - γ*ai
	elseif xi >= γ*bi
		return xi - γ*bi
	else
		return 0
	end
end
"""
    prox_boxconj(y,a,b) = (prox_i_conj(xi, ai, bi, γ))' i ∈ {1,..,n}

Compute the proximal operator of the convex conjugate of the indicator function
for the box contraint

	a <= x <= b

where the inequalites are applied element-wise.

"""
function prox_boxconj(y,a,b,γ)
	prox_i_conj.(y, a, b, γ)
end


"""
    dual2primal(y,Q,q,a,b)

Computes the solution to the primal problem for Hand-In 1 given a solution y to
the dual problem.
"""
function dual2primal(y,Q,q,a,b)
	return inv(Q)*(-y - q)
end
