# Imports. First the standard stuff
import numpy as np
import scipy.sparse as sps
from scipy.spatial.distance import cdist

from IPython.display import Image

import porepy as pp

# Also import the standardized model for poro-elastic contact problems. We will modify this model below.
import porepy.models.contact_mechanics_biot_model as model
def NNinterpolation( values, p, t):
    
    if p.shape[0] < p.shape[1]:
        p = np.transpose(p)
    if t.shape[0] < t.shape[1]:
        t = np.transpose(t)
    if len(values.shape) == 2:
        values = values[:,0]
    ''' Natural neighbor interpolation 
        Approximate values at nodes from cells integral'''
    indmat = np.zeros((t.shape[0],np.max(t) + 1  ))
    valmat = np.zeros((t.shape[0],np.max(t) + 1  ))         
    for e in range(t.shape[0] ):
        valmat[e,t[e,:]] = values[e]
        indmat[e,t[e,:]] = 1
    
    wei = np.ones((t.shape[0],1))
    valnod = np.dot(np.transpose(valmat),wei)/np.dot(np.transpose(indmat),wei) # values at interpolation points
    return valnod
def trisurf(p, t, value=None):
    import numpy as np
    if p.shape[0] < p.shape[1]:
        p = np.transpose(p)
    if t.shape[0] < t.shape[1]:
        t = np.transpose(t)
    import matplotlib.pyplot as plt
    fig, grid = plt.subplots()

    X = [p[t[:,0],0], p[t[:,1],0], p[t[:,2],0], p[t[:,0],0]]
    Y = [p[t[:,0],1], p[t[:,1],1], p[t[:,2],1], p[t[:,0],1]]
    grid.plot(X, Y, 'k-', linewidth = 0.2)
    
    if value is not None:
        if len(value.shape) == 1:
            value = value.reshape(len(value),1)
        name_color_map = 'jet'
        x = p[:,0]
        y = p[:,1]
        z = value[:,0]
        plt.tricontourf(x,y,t,z,1000,cmap = name_color_map)
        plt.colorbar()
    plt.show()
class BiotSetup(model.ContactMechanicsBiot):
    
    def __init__(self, params, mesh_args):
        """ Set arguments for mesh size (as explained in other tutorials)
        and the name fo the export folder.
        """
        # super().__init__(mesh_args, folder_name)
        # params kemur nú með folder_name
        super().__init__(params)
 
        # Set additional case specific fields
        # For now I take this out but can be used to set paremeters later on (see main.py (THM))
        self.mesh_args = mesh_args
        

        # File names for export
        # self.file_name = "biot"
        self.folder_name = self.params["folder_name"]
        self.file_name = self.params["file_name"]
        
        # Fix names used for the variables:
        # Names for mechanics variables are set in a super class
        self.scalar_variable = "p"
        self.mortar_scalar_variable = "mortar_" + self.scalar_variable
        
        self.scalar_coupling_term = "robin_" + self.scalar_variable
        self.scalar_parameter_key = "flow"

        self.subtract_fracture_pressure = True

        # Scaling coefficients
        # The use of this is not mature. Be careful!
        self.scalar_scale = 1e5
        self.length_scale = 100

        # Time 
        self.time = - pp.YEAR
        self.time_step = -self.time/2 #1500 * pp.DAY # obs scales with length scale
        self.end_time = self.time_step * 4

        self.initial_aperture = 1e-3 #/ self.length_scale
        # Dirichlet boundary condition for pressure
        self.s_0 = 1 * pp.MEGA / self.scalar_scale
        self.set_rock_and_fluid()
        # solution 
        self.disp = []
        self.pres = []

    def create_grid(self):
        """ Define a fracture network and domain and create a GridBucket. 
        
        This setup has two fractures inside the unit cell.
        
        The method also calls a submethod which sets injection points in 
        one of the fractures.
        
        """
        
        # Domain definition
        self.box = {"xmin": 0, "ymin": 0, "xmax": 1, "ymax": 1}
        # self.frac_pts = np.array([[0.2, 0.6, 0.4, 0.8],
        #                           [0.5, 0.4, 0.6, 0.5]])
        
        self.frac_pts = np.array([[0.2, 0.6, 0.4, 0.8],
                                  [0.3, 0.4, 0.6, 0.6]])
        frac_edges = np.array([[0, 2],
                               [1, 3]]) # Each column is one fracture

        network = pp.FractureNetwork2d(self.frac_pts, frac_edges, domain=self.box)
        # Generate the mixed-dimensional mesh
        gb = network.mesh(self.mesh_args)

        pp.contact_conditions.set_projections(gb)

        self.gb = gb
        self.Nd = self.gb.dim_max()
        self.well_cells()
        return gb

    def well_cells(self):
        """
        Assign unitary values to injection cells (positive) and production cells
        (negative).
        
        The wells are marked by g.tags['well_cells'], and in the data dictionary
        of this well.
        
        """
        for g, d in self.gb:
            tags = np.zeros(g.num_cells)
            if g.dim == self.Nd - 1:  # We should not ask for frac_num in intersections
                if g.frac_num == 4:
                    tags[1] = 1
            g.tags["well_cells"] = tags
            pp.set_state(d, {"well": tags.copy()})
    def bc_type_mechanics(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc = pp.BoundaryConditionVectorial(g, north + south, "dir")
        frac_face = g.tags["fracture_faces"]
        bc.is_neu[:, frac_face] = False
        bc.is_dir[:, frac_face] = True
        return bc
    def bc_values_mechanics(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros((g.dim, g.num_faces))
        # bc_values[0, north] = 0.001
        bc_values[1, north] = 0.001
        return bc_values.ravel("F")

    def bc_type_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        return pp.BoundaryCondition(g, west, "dir")

    def bc_values_scalar(self, g):
        all_bf, east, west, north, south, top, bottom = self.domain_boundary_sides(g)
        bc_values = np.zeros(g.num_faces)
        bc_values[west] = self.s_0
        return bc_values

    def source_scalar(self, g):
        values = np.zeros(g.num_cells)
        # if self.time > 1e-10:
        #     flow_rate = 0.001 * pp.MILLI * (pp.METER / self.length_scale) ** self.Nd
        #     values = flow_rate * g.tags["well_cells"]* self.time_step
        return values
    
    def source_mechanics(self, g):
        return np.zeros(g.num_cells * self.Nd)
        
    def compute_aperture(self, g):
        apertures = np.ones(g.num_cells)
        return apertures
    def set_permeability_from_aperture(self):
        """
        Cubic law in fractures, rock permeability in the matrix.
        """
        viscosity = self.fluid.dynamic_viscosity() / self.scalar_scale
        gb = self.gb
        key = self.scalar_parameter_key
        for g, d in gb:
            if g.dim < self.Nd:
                # Use cubic law in fractures
                apertures = self.compute_aperture(g)
                apertures_unscaled = apertures * self.length_scale
                k = np.power(apertures_unscaled, 2) / 12
                kxx = k / viscosity / self.length_scale ** 2
            else:
                # Use the rock permeability in the matrix
                kxx = (
                    self.rock.PERMEABILITY
                    / viscosity
                    * np.ones(g.num_cells)
                    / self.length_scale ** 2
                )           
            K = pp.SecondOrderTensor(kxx)
            d[pp.PARAMETERS][key]["second_order_tensor"] = K

        # Normal permeability inherited from fracture
        for e, d in gb.edges():
            mg = d["mortar_grid"]
            g_s, g_m = gb.nodes_of_edge(e)
            data_s = gb.node_props(g_s)
            a = self.compute_aperture(g_s)
            # We assume isotropic permeability in the fracture
            k_s = data_s[pp.PARAMETERS][self.scalar_parameter_key][
                "second_order_tensor"
            ].values[0, 0]
            kn = 2 * mg.slave_to_mortar_int() * np.divide(k_s, a)
            pp.initialize_data(mg, d, self.scalar_parameter_key, {"normal_diffusivity": kn})

    def set_rock_and_fluid(self):
        """
        Set rock and fluid properties to granite and water.
        """
        self.rock = pp.Granite()
        self.rock.FRICTION_COEFFICIENT = 0.5
        self.fluid = pp.Water()
        self.rock.PERMEABILITY = 1e-16

    def biot_alpha(self):
        return 0.8

    def set_mechanics_parameters(self):
        gb = self.gb
        for g, d in gb:
            if g.dim == self.Nd:
                # Rock parameters
                rock = self.rock
                lam = rock.LAMBDA * np.ones(g.num_cells) / self.scalar_scale
                mu = rock.MU * np.ones(g.num_cells) / self.scalar_scale
                C = pp.FourthOrderTensor(mu, lam)
            
                bc = self.bc_type_mechanics(g)

                bc_values = self.bc_values_mechanics(g)
                sources = self.source_mechanics(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {
                        "bc": bc,
                        "bc_values": bc_values,
                        "source": sources,
                        "fourth_order_tensor": C,
                        "biot_alpha": self.biot_alpha()
                    },
                )
                d[pp.PARAMETERS].set_from_other(
                    self.mechanics_parameter_key,
                    self.scalar_parameter_key,
                    ["aperture", "time_step"],
                )
            elif g.dim == self.Nd - 1:
                friction = self._set_friction_coefficient(g)
                pp.initialize_data(
                    g,
                    d,
                    self.mechanics_parameter_key,
                    {"friction_coefficient": friction},
                )
                pp.initialize_data(g, d, self.mechanics_parameter_key, {})

        for e, d in gb.edges():
            mg = d["mortar_grid"]
            # Parameters for the surface diffusion. No clue about values
            mu = self.rock.MU
            lmbda = self.rock.LAMBDA
            pp.initialize_data(
                mg, d, self.mechanics_parameter_key, {"mu": mu, "lambda": lmbda}
            )

    def set_scalar_parameters(self):
         # * self.scalar_scale
        for g, d in self.gb:
            # Aperture [m] and cross sectional area [m**(self.Nd - g.dim)]
            a = self.compute_aperture(g)
            cross_sectional_area = np.power(a, self.gb.dim_max() - g.dim) * np.ones(
                g.num_cells
            )
            # Define boundary conditions for flow
            bc = self.bc_type_scalar(g)

            # Set boundary condition values
            bc_val = self.bc_values_scalar(g)

            # and source values
            sources = self.source_scalar(g)
            alpha = self.biot_alpha()

            # Add fluid flow dictionary
            pp.initialize_data(
                g,
                d,
                self.scalar_parameter_key,
                {
                    "bc": bc,
                    "bc_values": bc_val,
                    "mass_weight": self.fluid.COMPRESSIBILITY,
                    "aperture": cross_sectional_area,
                    "time_step": self.time_step,
                    "source": sources,
                    "biot_alpha": alpha,
                },
            )
        for _, d in self.gb.edges():
            mg = d["mortar_grid"]
            pp.initialize_data(mg, d, self.scalar_parameter_key)
        self.set_permeability_from_aperture()


    def initial_condition(self, time=0):
        """
        Initial guess for Newton iteration.
        """
        gb = self.gb
        key_m = self.mechanics_parameter_key

        for g, d in gb:
            nc_nd = g.num_cells * self.Nd
            # Initial value for the scalar variable.
            initial_scalar_value = self.s_0 * np.ones(g.num_cells)

            if g.dim == self.Nd:
                initial_displacements = np.zeros(nc_nd)
                # Initialize displacement variable
                state = {
                    self.displacement_variable: initial_displacements,
                    self.scalar_variable: initial_scalar_value,
                    key_m: {"bc_values": d[pp.PARAMETERS][key_m]["bc_values"]},
                }
            elif g.dim == self.Nd - 1:
                # Initialize contact variable
                traction = np.vstack(
                    (
                        np.zeros((self.Nd - 1, g.num_cells)),
                        -100 * np.ones(g.num_cells),
                    )
                ).ravel(order="F")
                state = {
                        self.contact_traction_variable: traction,
                        "previous_iterate": {self.contact_traction_variable: traction},
                        self.scalar_variable: initial_scalar_value,
                        }
            pp.set_state(d, state)
            
        for e, d in gb.edges():
            mg = d["mortar_grid"]
        
            initial_displacements = np.zeros(mg.num_cells * self.Nd)
                
            if mg.dim == self.Nd - 1:
                state = {
                    self.mortar_scalar_variable: np.zeros(mg.num_cells),
                    self.mortar_displacement_variable: initial_displacements,
                    "previous_iterate": {
                        self.mortar_displacement_variable: initial_displacements
                    },
                }
                pp.set_state(d, state)

    def after_newton_convergence(self, solution, errors, iteration_counter):
        super().after_newton_convergence(solution, errors, iteration_counter)
        self.export_step()
        dispi, presi = self.export_results()
        self.disp.append(dispi)
        self.pres.append(presi)
        
    def _set_friction_coefficient(self, g):
        nodes = g.nodes
        tips = nodes[:, [0, -1]]
        fc = g.cell_centers
        D = cdist(fc.T, tips.T)
        D = np.min(D, axis=1)
        R = 200
        beta = 10
        friction_coefficient = self.rock.FRICTION_COEFFICIENT * (1 + beta * np.exp(-R * D ** 2))
        return friction_coefficient

    def set_exporter(self):
        self.exporter = pp.Exporter(self.gb, self.file_name, folder_name=self.folder_name)
        self.export_fields = ["u_exp", "p_exp", "well"]
        self.export_times = []
        

    def export_step(self):
        if "exporter" not in self.__dict__:
            self.set_exporter()
        for g, d in self.gb:
            if g.dim == self.Nd:
                u = d[pp.STATE][self.displacement_variable].reshape(
                    (setup.Nd, -1), order="F"
                )
                if g.dim == 3:
                    d[pp.STATE]["u_exp"] = u * setup.length_scale
                else:
                    d[pp.STATE]["u_exp"] = np.vstack(
                        (u * setup.length_scale, np.zeros(u.shape[1]))
                    )
            else:
                g_h = self.gb.node_neighbors(g)[0]
                assert g_h.dim == self.Nd
                data_edge = self.gb.edge_props((g, g_h))
                u_mortar_local = self.reconstruct_local_displacement_jump(
                    data_edge,
                )
                if g.dim == 2:
                    d[pp.STATE]["u_exp"] = u_mortar_local * setup.length_scale
                else:
                    d[pp.STATE]["u_exp"] = np.vstack(
                        (
                            u_mortar_local * setup.length_scale,
                            np.zeros(u_mortar_local.shape[1]),
                        )
                    )
            d[pp.STATE]["p_exp"] = d[pp.STATE][self.scalar_variable] * self.scalar_scale
        self.exporter.write_vtk(self.export_fields, time_step=self.time)
        self.export_times.append(self.time)

    def export_results(self):
        """
        Save displacement jumps and number of iterations for visualisation purposes. 
        These are written to file and plotted against time in Figure 4.
        """     
        g = self.gb.grids_of_dimension(2)[0]
        data = self.gb.node_props(g)
        disp = data[pp.STATE]["u"]
        pres = data[pp.STATE]["p"]
        return disp, pres
   
Nd = 2
mesh_size = 0.02
#mesh_args = {mesh_size, 0.1*mesh_size, 3*mesh_size}

mesh_args = {
   "mesh_size_frac": mesh_size,
    "mesh_size_min": 0.1 * mesh_size,
    "mesh_size_bound": 3 * mesh_size,
}

params = {
    "folder_name": "biot_2",
    "convergence_tol": 2e-7,  # Var 2e-7
    "max_iterations": 20,
    "file_name": "main_run",
}

setup = BiotSetup(params, mesh_args)
pp.run_time_dependent_model(setup, params)

gb = setup.create_grid()
g = gb.grids_of_dimension(2)[0]
t = g.cell_nodes().indices.reshape((3, g.num_cells), order='f')
p = g.nodes
p = p[[0,1],:]
# pp.plot_grid(g, pressure[-1])
for i in range(6):
    displacement = setup.disp[i]
    pressure = setup.pres[i]
    
    presi = NNinterpolation( pressure, p, t)
    
    uxc = displacement[0::2]
    uyc = displacement[1::2]
    uxi = NNinterpolation( uxc, p, t)
    uyi = NNinterpolation( uyc, p, t)
    
    defor = np.concatenate((uxi,uyi), axis = 1)
    
    trisurf(np.transpose(p) + defor*1e2, t, presi)





