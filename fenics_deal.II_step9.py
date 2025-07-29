#!/mnt/d/software_install/fenics-0.9/bin/python3
from mpi4py import MPI
import numpy as np
from dolfinx.fem.petsc import LinearProblem
import dolfinx, ufl, basix
import pyvista
from dolfinx import plot
from dolfinx import cpp as _cpp
from dolfinx.cpp.mesh import GhostMode
ghost_mode = GhostMode.shared_facet

N = 8
domain = dolfinx.mesh.create_rectangle(
    comm=MPI.COMM_WORLD,
    points=[(-1.0, -1.0), (1.0, 1.0)],
    n = [N, N],
    ghost_mode= ghost_mode)

N_refine = 8
for i_iter in range(N_refine):
    print(f"++++++++{i_iter}++++++++")
    V = dolfinx.fem.functionspace(domain, ("Lagrange", 1))
    x = ufl.SpatialCoordinate(domain)
    
    # 速度场
    beta = ufl.as_vector([2.0, 
                          1 + 4.0/5.0*ufl.sin(8.0*ufl.pi*x[0])])
    # 面元的法向量
    n = ufl.FacetNormal(domain)

    # 在fenics 中虽然可以设置meshtag 来选取边界，但使用 dolfinx.fem.locate_dofs_geometrical时，
    # marker函数接受的参数只有点的坐标，得不到法向量信息。这里我们把 β·n 看作一个函数
    # 让它在流出边界上取值为0，然后把该函数放进弱形式中
    beta_dot_n = ufl.dot(beta, n)
    beta_dot_n_inflow = ufl.conditional(ufl.lt(beta_dot_n, 0), beta_dot_n, 0.0)

    x0 = [-3.0/4, -3.0/4]
    delta = 0.1 * ufl.Circumradius(domain)
    s = 0.1
    f = ufl.conditional(
        ufl.lt((x[0]-x0[0])**2 + (x[1]-x0[1])**2, s*s), 
        1.0/(10*s*s), 
        0)

    norm_x2 = x[0]**2 + x[1]**2
    g = ufl.exp(5*(1-norm_x2)) * ufl.sin(16.0*ufl.pi*norm_x2)

    u = ufl.TrialFunction(V)
    v = ufl.TestFunction(V)

    # 双线性形式
    a = ufl.dot(beta, ufl.grad(u)) * (v + delta*ufl.dot(beta, ufl.grad(v))) * ufl.dx - u*beta_dot_n_inflow*v*ufl.ds
    # 线性形式
    L = f*(v + delta*ufl.dot(beta, ufl.grad(v)))*ufl.dx - g*beta_dot_n_inflow*v*ufl.ds

    problem = LinearProblem(a, L, bcs=[], 
                            petsc_options={"ksp_type": "gmres", 
                                "pc_type":  "lu", 
                                "ksp_converged_reason":"",
                                "ksp_monitor":""})
    uh = problem.solve()

    filename = f"solution_iter_{i_iter}.xdmf"
    with dolfinx.io.XDMFFile(domain.comm, filename, "w") as xdmf:
        xdmf.write_mesh(domain)
        xdmf.write_function(uh)
    
    # if i_iter == 8:
    #     tdim = domain.topology.dim
    #     domain.topology.create_connectivity(tdim, tdim)
    #     topology, cell_types, geometry = plot.vtk_mesh(domain, tdim)
    #     grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    #     plotter = pyvista.Plotter()
    #     plotter.add_mesh(grid, show_edges=True)
    #     plotter.view_xy()
    #     plotter.show()

    c_dim = domain.geometry.dim
    ###### 获取每个cell的中心 ######
    # 0阶的非连续Galerkin方法的插值点就在cell的中心
    # 把有限元的形状设置成mesh的几何维度，从而可以存放cell中心的坐标
    Q = dolfinx.fem.functionspace(domain, ("DG", 0, (c_dim,)))
    # 使用interpolate后，q.x.array中就存放了cell中心的坐标
    q = dolfinx.fem.Function(Q)
    q.interpolate(lambda x: x[:c_dim])
    # q_r 是cell中心的坐标
    q_r = q.x.array.reshape(-1, c_dim)

    ###### 获取所有不在边界上的边的索引 ###### 
    num_cells_local = domain.topology.index_map(c_dim).size_local \
                + domain.topology.index_map(c_dim).num_ghosts

    domain.topology.create_connectivity(c_dim - 1, c_dim)
    domain.topology.create_connectivity(c_dim, c_dim - 1)
    exterior_facet_indices = dolfinx.mesh.exterior_facet_indices(domain.topology)
    num_facets_local = domain.topology.index_map(domain.topology.dim - 1).size_local
    interior_facet_indices = np.arange(num_facets_local)
    interior_facet_indices = np.delete(interior_facet_indices, exterior_facet_indices)

    ###### 获取cell和facet之间的连接关系 ###### 
    f_to_c = domain.topology.connectivity(c_dim - 1, c_dim)

    ###### 获取所有连接相邻两个cell中心的边，这里的索引实际上是cell ###### 
    edge_index = [f_to_c.links(interior_facet) for interior_facet in interior_facet_indices]
    #print(edge_index)
    edge_index = np.vstack(edge_index)
    edge_index = np.vstack((edge_index, edge_index[:, ::-1])).T

    y_kk = q_r[edge_index[0]] - q_r[edge_index[1]]
    L_y_kk = np.linalg.norm(y_kk, axis=1)[:, None]
    y_kk = y_kk / L_y_kk #向量y_kk已经归一化
    
    # 获得reference cell的顶点坐标，对于三角形元就是[0,0], [0,1], [1,0]
    ref_cell_geometry = basix.cell.geometry(domain.basix_cell())
    # 然后求中心点
    midpoint = np.sum(ref_cell_geometry, axis=0) / ref_cell_geometry.shape[0]

    uh_at_midpoint = dolfinx.fem.Expression(uh, midpoint.reshape(-1, len(midpoint)))
    # (num_cells_local, 1)
    midpoint_values = uh_at_midpoint.eval(domain, np.arange(num_cells_local, dtype=np.int32))
    
    h_local = dolfinx.fem.Expression(ufl.Circumradius(domain), midpoint.reshape(-1, len(midpoint)))
    # (num_cells_local, 1)
    h_values = h_local.eval(domain, np.arange(num_cells_local, dtype=np.int32))

    d_uh = midpoint_values[edge_index[0]] - midpoint_values[edge_index[1]]
    grad_uh = y_kk * d_uh / L_y_kk

    grad_uh_cell = []
    for i in range(num_cells_local):
        index = np.where(edge_index[0] == i)[0]
        g = grad_uh[index].mean(axis=0)
        y_kk_sub = y_kk[index]
        Y = np.sum(np.einsum('bi, bj->bij', y_kk_sub, y_kk_sub), axis=0)
        if np.linalg.det(Y) < 1.0e-6:
            grad_uh_cell.append(100.0)
        else:
            grad_cell = np.matmul(np.linalg.inv(Y), g) 
            grad_cell = np.linalg.norm(grad_cell) * (h_values[i]**2)
            grad_uh_cell.append(grad_cell[0])
    
    grad_uh_cell = np.array(grad_uh_cell)

    sort_index = np.argsort(grad_uh_cell)

    cell_index = sort_index[-int(0.333*num_cells_local):]

    edge_index = dolfinx.mesh.compute_incident_entities(domain.topology, cell_index.astype(np.int32), 2, 1)

    partitioner = _cpp.mesh.create_cell_partitioner(ghost_mode)

    new_domain, _, _ = dolfinx.mesh.refine(domain, edge_index, partitioner=partitioner)
    domain = new_domain