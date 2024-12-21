### **1. Conservation of Mass (Continuity Equation)**

The conservation of mass states that the total mass in a closed system remains constant over time. For fluids, it can be expressed as follows:

#### **Integral Form**
\[
\frac{d}{dt} \int_V \rho \, dV + \int_{\partial V} \rho \mathbf{v} \cdot \mathbf{n} \, dA = 0
\]
where:
- \(\rho\) is the fluid density,
- \(V\) is the control volume,
- \(\partial V\) is the boundary of the control volume,
- \(\mathbf{v}\) is the fluid velocity,
- \(\mathbf{n}\) is the unit normal vector to the boundary.

#### **Differential Form**
Applying the divergence theorem and shrinking the control volume to an infinitesimal point yields the differential form:
\[
\frac{\partial \rho}{\partial t} + \nabla \cdot (\rho \mathbf{v}) = 0
\]
where:
- \(\frac{\partial \rho}{\partial t}\) represents the local rate of change of density,
- \(\nabla \cdot (\rho \mathbf{v})\) represents the net mass flux.

For incompressible fluids (\(\rho = \text{constant}\)), this simplifies to:
\[
\nabla \cdot \mathbf{v} = 0
\]

---

### **2. Conservation of Momentum (Navier-Stokes Equations)**

The conservation of momentum, derived from Newton's second law, states that the rate of change of momentum is equal to the sum of forces acting on the fluid.

#### **Integral Form**
\[
\frac{d}{dt} \int_V \rho \mathbf{v} \, dV + \int_{\partial V} \rho \mathbf{v} (\mathbf{v} \cdot \mathbf{n}) \, dA = \int_V \mathbf{f} \, dV + \int_{\partial V} \mathbf{T} \, dA
\]
where:
- \(\rho \mathbf{v}\) is the momentum per unit volume,
- \(\mathbf{f}\) is the body force (e.g., gravity),
- \(\mathbf{T}\) is the surface stress tensor.

#### **Differential Form**
Using the divergence theorem and shrinking the control volume leads to the differential form:
\[
\rho \frac{\partial \mathbf{v}}{\partial t} + \rho (\mathbf{v} \cdot \nabla) \mathbf{v} = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{f}
\]
where:
- \(\rho \frac{\partial \mathbf{v}}{\partial t}\) is the local rate of change of momentum,
- \(\rho (\mathbf{v} \cdot \nabla) \mathbf{v}\) is the convective momentum flux,
- \(-\nabla p\) is the pressure gradient force,
- \(\mu \nabla^2 \mathbf{v}\) represents viscous forces (\(\mu\) is the dynamic viscosity),
- \(\mathbf{f}\) is the body force per unit volume.

For steady and incompressible fluids, the equation simplifies to:
\[
\rho (\mathbf{v} \cdot \nabla) \mathbf{v} = -\nabla p + \mu \nabla^2 \mathbf{v} + \mathbf{f}
\]

---

These two differential equations form the foundation of fluid dynamics. The continuity equation governs the mass distribution of the fluid, while the momentum equation governs the motion of the fluid under the influence of forces.