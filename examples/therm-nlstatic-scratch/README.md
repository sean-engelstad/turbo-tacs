# Write a nonlinear static analysis myself, not using code from TACSContinuation.h, although can be inspired from it
Sean Engelstad

* Noticed the nonlinear static analysis for the elastic elements with pure temperature driven loading doesn't make sense.
Namely, with perfect geometry => the local eigenvalues don't match the initial linear buckling about zero disps predictions (while
it matched to high precision with the mechanical buckling on a perfect cylinder). For imperfections, the eigenvalues are much
smaller than you would think when in the nonlinear regime. I think the linear buckling solution about zero disps is fine.
Suspect something might be wrong with the nonlinear static solution for thermal buckling, so going to code it up myself from scratch
so that I can play with it (for perfect geometry first).
* Here's how I plan to code it up: just going to do Newton solves since I am just scaling the temperatures which is similar to disp control.
And disp control doesn't really require arc length method unless you loop backwards on yourself (then you do). Should be ok I think.

Consider disps vector u, loading factor lambda and the nonlinear residual R = dPi/du. We wish to solve for the minimum energy state.
Suppose we have solved up to u,lambda state and we want to increase the temps to some new load factor lambda' = lambda + dlambda.
We now have the residual equation:

R(u', lambda') = R(u+du, lambda+dlambda) = 0

Linearize this equation by changing u only (prescribed lambda' now). Could linearize about lambda as well, but we know what it is and can plug it in
so let's not do that. This is not like load control where we don't know how much to increase the loads because we might change from inc the load to dec it.

R(u',lambda') approx R(u,lambda') + [dR/du](u,lambda') * du = 0

Letting K_t = dR/du the tangent stiffness matrix and since we know R(u,lambda') [can plug it in], we get:

K_t(u,lambda') * du = - R(u,lambda')

This is our linear solve now! So plan to increase lambda => lambda + dlambda in load factor increments (increasing the magnitude of the prescribed temp field)
and completing this linear solve each time, updating u. Since we do predict buckling with the linear buckling about zero disps, I expect that if we do this
right, it should experience stress-stiffening where the stiffness decreases with increasing compressive load. Let's try it from scratch!